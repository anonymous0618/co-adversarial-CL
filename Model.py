import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args

# args.user = 123
# args.item = 423
xavierInit = nn.init.xavier_uniform_
zeroInit = lambda x: nn.init.constant_(x, 0.0)
normalInit = lambda x: nn.init.normal_(x, 0.0, 0.3)

class BiasMF(nn.Module):
	def __init__(self):
		super(BiasMF, self).__init__()

		self.uEmbeds = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
		self.uBias = nn.Parameter(zeroInit(t.empty(args.user)))
		self.iBias = nn.Parameter(zeroInit(t.empty(args.item)))

	def forward(self):
		return (self.uEmbeds), (self.iEmbeds), self.uBias, self.iBias

	def predPairs(self, usr, itm):
		uEmbed = (self.uEmbeds[usr])
		iEmbed = (self.iEmbeds[itm])
		uBias = self.uBias[usr]
		iBias = self.iBias[itm]
		return t.sum(uEmbed * iEmbed, axis=-1).view(-1) + uBias + iBias

	def predAll(self, usr):
		uEmbed = (self.uEmbeds[usr])
		iEmbeds = (self.iEmbeds)
		uBias = self.uBias[usr].view([-1, 1])
		iBias = self.iBias.view(1, -1)
		return t.mm(uEmbed, t.transpose(iEmbeds, 1, 0)) + uBias + iBias

	def predBatch(self, usr, itm):
		uEmbed = (self.uEmbeds[usr].view([-1, 1, args.latdim]))
		iEmbeds = (t.transpose(self.iEmbeds[itm], 1, 2))
		preds = t.squeeze(t.bmm(uEmbed, iEmbeds))
		uBias = self.uBias[usr].view([-1, 1])
		iBias = self.iBias[itm]
		preds = preds + uBias + iBias
		return preds

class LightGCN(nn.Module):
	def __init__(self, uEmbeds=None, iEmbeds=None):
		super(LightGCN, self).__init__()

		self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
		self.iEmbeds = iEmbeds if iEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

	def forward(self, adj):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)
		return embeds[:args.user], embeds[args.user:]

	def predPairs(self, adj, usr, itm):
		uEmbeds, iEmbeds = self.forward(adj)
		uEmbed = uEmbeds[usr]
		iEmbed = iEmbeds[itm]
		return t.sum(uEmbed * iEmbed, axis=-1).view(-1)

	def predAll(self, adj, usr, itm=None):
		uEmbeds, iEmbeds = self.forward(adj)
		uEmbed = uEmbeds[usr]
		if itm is not None:
			iEmbeds = iEmbeds[itm]
		return t.mm(uEmbed, t.transpose(iEmbeds, 1, 0))

	def predBatch(self, adj, usr, itm):
		uEmbeds, iEmbeds = self.forward(adj)
		uEmbed = uEmbeds[usr].view([-1, 1, args.latdim])
		iEmbeds = t.transpose(iEmbeds[itm], 1, 2)
		preds = t.squeeze(t.bmm(uEmbed, iEmbeds))
		return preds

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()
		# self.weight = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)

class HGNN(nn.Module):
	def __init__(self):
		super(HGNN, self).__init__()

		self.uEmbeds = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
		self.uHyperEmbeds = nn.Parameter(xavierInit(t.empty(args.latdim, args.hyperNum)))
		self.iHyperEmbeds = nn.Parameter(xavierInit(t.empty(args.latdim, args.hyperNum)))
		self.uHgnnLayers = nn.Sequential(*[HGNNLayer() for i in range(args.gnn_layer)])
		self.iHgnnLayers = nn.Sequential(*[HGNNLayer() for i in range(args.gnn_layer)])

		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		self.SpAdjDropEdge = SpAdjDropEdge(args.keepRate)

	def forward(self, adj):
		uEmbedsLst = [self.uEmbeds]
		iEmbedsLst = [self.iEmbeds]
		uuHyper = t.mm(self.uEmbeds, self.uHyperEmbeds)
		iiHyper = t.mm(self.iEmbeds, self.iHyperEmbeds)
		for i in range(args.gnn_layer):
			allEmbeds = t.concat([uEmbedsLst[-1], iEmbedsLst[-1]], axis=0)
			allEmbeds = self.gcnLayers[i](self.SpAdjDropEdge(adj), allEmbeds)
			uEmbeds, iEmbeds = allEmbeds[:args.user], allEmbeds[args.user:]
			uHyperEmbeds = self.uHgnnLayers[i](uEmbedsLst[-1], uuHyper)
			iHyperEmbeds = self.iHgnnLayers[i](iEmbedsLst[-1], iiHyper)
			uEmbedsLst.append(uEmbeds + uHyperEmbeds)# + uEmbedsLst[-1])
			iEmbedsLst.append(iEmbeds + iHyperEmbeds)# + iEmbedsLst[-1])
		return sum(uEmbedsLst), sum(iEmbedsLst)

	def predPairs(self, adj, usr, itm):
		uEmbeds, iEmbeds = self.forward(adj)
		uEmbed = uEmbeds[usr]
		iEmbed = iEmbeds[itm]
		return t.sum(uEmbed * iEmbed, axis=-1).view(-1)

	def predAll(self, adj, usr, itm=None):
		uEmbeds, iEmbeds = self.forward(adj)
		uEmbed = uEmbeds[usr]
		if itm is not None:
			iEmbeds = iEmbeds[itm]
		return t.mm(uEmbed, t.transpose(iEmbeds, 1, 0))

	def predBatch(self, adj, usr, itm):
		uEmbeds, iEmbeds = self.forward(adj)
		uEmbed = uEmbeds[usr].view([-1, 1, args.latdim])
		iEmbeds = t.transpose(iEmbeds[itm], 1, 2)
		preds = t.squeeze(t.bmm(uEmbed, iEmbeds))
		return preds

class HGNNLayer(nn.Module):
	def __init__(self):
		super(HGNNLayer, self).__init__()

		# self.hyperEmbeds = hyperEmbeds
		self.act = nn.LeakyReLU(0.5)

	def forward(self, embeds, hyper):
		hyperEmbeds = self.act(t.mm(t.transpose(embeds, 0, 1), hyper))
		hyperTp = t.transpose(hyper, 0, 1)
		nodeEmbeds = self.act(t.transpose(t.mm(hyperEmbeds, hyperTp), 0, 1))
		return nodeEmbeds

class SpAdjDropEdge(nn.Module):
	def __init__(self, keepRate):
		super(SpAdjDropEdge, self).__init__()

		self.keepRate = keepRate

	def forward(self, adj):
		vals = adj._values()
		idxs = adj._indices()
		edgeNum = vals.size()
		mask = ((t.rand(edgeNum) + self.keepRate).floor()).type(t.bool)
		newVals = vals[mask] / self.keepRate
		newIdxs = idxs[:, mask]
		return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)

class EdgeHGNN(nn.Module):
	def __init__(self):
		super(EdgeHGNN, self).__init__()

		self.uEmbeds = nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
		self.edgeTrans = nn.Parameter(xavierInit(t.empty(args.eTypeNum, args.latdim, args.latdim)))
		# self.queryLinear =nn.Linear(args.latdim, args.latdim)
		# self.hyperEmbeds = nn.Parameter(xavierInit(t.empty(args.latdim, args.hyperNum)))
		self.HGNNLayers = nn.Sequential(*[HGNNLayer() for i in range(args.gnn_layer)])

	def forward(self, adj):
		src = adj._indices()[0, :]
		tgt = adj._indices()[1, :]
		val = adj._values()
		srcEmbeds = self.uEmbeds[src].view([-1, 1, args.latdim])
		tgtEmbeds = self.iEmbeds[tgt]
		edgeTrans = self.edgeTrans[val]
		edgeEmbeds = t.squeeze(t.bmm(srcEmbeds, edgeTrans)) * tgtEmbeds
		for i in range(args.gnn_layer):
			edgeEmbeds = self.HGNNLayers[i](edgeEmbeds) + edgeEmbeds
		sptEmbeds = t.sparse.FloatTensor(adj._indices(), edgeEmbeds, adj.shape)
		srcEmbeds = t.sparse.sum(sptEmbeds, 1)
		tgtEmbeds = t.sparse.sum(sptEmbeds, 0)
		return srcEmbeds, tgtEmbeds