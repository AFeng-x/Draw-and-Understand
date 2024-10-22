__author__ = 'tylin'
from .tokenizer.ptbtokenizer import PTBTokenizer
from .meteor.meteor import Meteor
from .cider.cider import Cider
import pdb
class COCOEvalCap:
    def __init__(self):
        self.evalImgs = []
        self.eval = {}
        self.imgToEval = {}
        # self.coco = coco
        # self.cocoRes = cocoRes
        # self.params = {'image_id': coco.getImgIds()}

    def evaluate(self, gts, res):
        imgIds = []
        for key in gts.keys():
            imgIds.append(int(key))
        # imgIds = self.coco.getImgIds()
        # gts = {}
        # res = {}
        # for imgId in imgIds:
        #     gts[imgId] = self.coco.imgToAnns[imgId]
        #     res[imgId] = self.cocoRes.imgToAnns[imgId]
        '''
        gts:
        483234: [
            {
                u'image_id': 483234, 
                u'id': 773888, 
                u'caption': u'A person is standing in the snow near a tree with skis and snowboards. '
            }, {u'image_id': 483234, u'id': 774641, u'caption': u'Snowboards resting upon a tree, with man hiding inside it like fort'}, {u'image_id': 483234, u'id': 775001, u'caption': u'A person holds a snowboard in front of a tree with snowboards leaning on it.'}, {u'image_id': 483234, u'id': 775055, u'caption': u'a person holding some skis walking through the snow '}, {u'image_id': 483234, u'id': 776300, u'caption': u'a a couple of snowboards are up against a tree'}]

        res:
        483234: [
            {
                u'image_id': 483234, 
                'id': 531, 
                u'caption': u'man standing on top of a snow covered slope'
            }
        ]
        '''
        # pdb.set_trace()
        # =================================================
        # Set up scorers
        # =================================================
        print('tokenization...')
        tokenizer = PTBTokenizer()
        gts  = tokenizer.tokenize(gts)
        res = tokenizer.tokenize(res)

        # =================================================
        # Set up scorers
        # =================================================
        print('setting up scorers...')
        scorers = [
            (Meteor(),"METEOR"),
            (Cider(), "CIDEr")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print ('computing %s score...'%(scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    self.setEval(sc, m)
                    self.setImgToEvalImgs(scs, gts.keys(), m)
                    print ("%s: %0.3f"%(m, sc))
            else:
                self.setEval(score, method)
                self.setImgToEvalImgs(scores, gts.keys(), method)
                print("%s: %0.3f"%(method, score))
        self.setEvalImgs()

    def setEval(self, score, method):
        self.eval[method] = score

    def setImgToEvalImgs(self, scores, imgIds, method):
        for imgId, score in zip(imgIds, scores):
            if not imgId in self.imgToEval:
                self.imgToEval[imgId] = {}
                self.imgToEval[imgId]["image_id"] = imgId
            self.imgToEval[imgId][method] = score

    def setEvalImgs(self):
        self.evalImgs = [eval for imgId, eval in self.imgToEval.items()]