import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import BiasMF, LightGCN, HGNN
from DataHandler import DataHandler
import numpy as np
import pickle

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', len(self.handler.trnLoader.dataset.rows))
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			self.prepareModel()
			log('Model Prepared')
		for ep in range(stloc, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, tstFlag))
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def prepareModel(self):
		self.LightGCN = LightGCN().cuda()
		self.LightGCN2 = self.LightGCN#LightGCN().cuda()
		# self.BiasMF = BiasMF().cuda()
		# self.HGNN = HGNN().cuda()
		self.gcnOpt = t.optim.Adam(self.LightGCN.parameters(), lr=args.lr, weight_decay=0)
		self.gcnOpt2 = t.optim.Adam(self.LightGCN2.parameters(), lr=args.lr, weight_decay=0)
		# self.mfOpt = t.optim.Adam(self.BiasMF.parameters(), lr=args.lr, weight_decay=0)
		# self.hgnnOpt = t.optim.Adam(self.HGNN.parameters(), lr=args.lr, weight_decay=0)

	def trainUtil(self, model, opt, usr, itmP, itmN, params=list()):
		predsP = model.predPairs(*(params + [usr, itmP]))
		predsN = model.predPairs(*(params + [usr, itmN]))
		scoreDiff = predsP - predsN
		bprLoss = - scoreDiff.sigmoid().log().sum() / args.batch
		regLoss = 0
		for W in model.parameters():
			regLoss += W.norm(2).square()
		regLoss *= args.reg
		loss = bprLoss + regLoss
		opt.zero_grad()
		loss.backward()
		opt.step()
		return loss, bprLoss

	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epMfLoss, epMfPreLoss, epGcnLoss, epGcnPreLoss = [0] * 4
		i = 0
		steps = trnLoader.dataset.__len__() // args.batch
		for usr, itmP, itmN, advNegs in trnLoader:
			i += 1
			usr = usr.long().cuda()
			itmP = itmP.long().cuda()
			itmN = itmN.long().cuda()
			advNegs = advNegs.long().cuda()

			# train mf
			mfLoss, mfPreLoss = self.trainUtil(self.LightGCN2, self.gcnOpt2, usr, itmP, itmN, [self.handler.torchAdj])
			epMfLoss += mfLoss.item()
			epMfPreLoss += mfPreLoss.item()

			# train lightgcn
			gcnLoss, gcnPreLoss = self.trainUtil(self.LightGCN, self.gcnOpt, usr, itmP, itmN, [self.handler.torchAdj])
			epGcnLoss += gcnLoss.item()
			epGcnPreLoss += gcnPreLoss.item()

			# MF generate, GCN discriminate
			batPreds = self.LightGCN.predBatch(self.handler.torchAdj, usr, advNegs)
			_, topLocs = t.topk(batPreds, 1)
			itmG = t.gather(advNegs, 1, topLocs).view(-1)
			gcnLoss, gcnPreLoss = self.trainUtil(self.LightGCN, self.gcnOpt, usr, itmP, itmG, [self.handler.torchAdj])
			epGcnLoss += gcnLoss.item()
			epGcnPreLoss += gcnPreLoss.item()
			gcnLoss, gcnPreLoss = self.trainUtil(self.LightGCN, self.gcnOpt, usr, itmG, itmN, [self.handler.torchAdj])
			epGcnLoss += gcnLoss.item()
			epGcnPreLoss += gcnPreLoss.item()

			# GCN generate, MF discriminate
			batPreds = self.LightGCN.predBatch(self.handler.torchAdj, usr, advNegs)
			_, topLocs = t.topk(batPreds, 1)
			itmG = t.gather(advNegs, 1, topLocs).view(-1)
			mfLoss, mfPreLoss = self.trainUtil(self.LightGCN2, self.gcnOpt2, usr, itmP, itmG, [self.handler.torchAdj])
			epMfLoss += mfLoss.item()
			epMfPreLoss += mfPreLoss.item()
			mfLoss, mfPreLoss = self.trainUtil(self.LightGCN2, self.gcnOpt2, usr, itmG, itmN, [self.handler.torchAdj])
			epMfLoss += mfLoss.item()
			epMfPreLoss += mfPreLoss.item()

			log('Step %d/%d         ' % (i, steps), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epGcnLoss / steps
		ret['preLoss'] = epGcnPreLoss / steps
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader
		epLoss, epRecall, epNdcg = [0] * 3
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat
		for usr, trnMask in tstLoader:
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()

			allPreds = self.LightGCN.predAll(self.handler.torchAdj, usr)
			allPreds = allPreds * (1 - trnMask) - trnMask * 1e8

			_, topLocs = t.topk(allPreds, args.topk)
			recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
			epRecall += recall
			epNdcg += ndcg
			log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
		ret = dict()
		ret['Recall'] = epRecall / num
		ret['NDCG'] = epNdcg / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		recallBig = 0
		ndcgBig =0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		content = {
			# 'BiasMF': self.BiasMF,
			'LightGCN': self.LightGCN,
			'LightGCN2': self.LightGCN2,
			# 'HGNN': self.HGNN
		}
		t.save(content, 'Models/' + args.save_path + '.mod')
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		ckp = t.load('Models/' + args.load_model + '.mod')
		# self.BiasMF = ckp['BiasMF']
		self.LightGCN = ckp['LightGCN']
		self.LightGCN2 = ckp['LightGCN2']
		# self.HGNN = ckp['HGNN']

		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		self.gcnOpt = t.optim.Adam(self.LightGCN.parameters(), lr=args.lr, weight_decay=0)
		self.gcnOpt2 = t.optim.Adam(self.LightGCN2.parameters(), lr=args.lr, weight_decay=0)
		log('Model Loaded')	

if __name__ == '__main__':
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()