# cloth_changing_person_ReID


os block https://github.com/MatthewAbugeja/osnet/blob/master/torchreid/models/osnet.py
https://arxiv.org/pdf/1910.06827v5

189 三个并行结构进行判别
0.17963
0.42602

190 Multi-granularity Cross Transformer Network for person re-identification
0.17243
0.375

191  return F.relu(out) + identity
0.17184
0.37755

192 修改att [BUG]
https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py

193 修改att
https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/CBAM.py
0.17702
0.39031

194 平均池化
output = self.sigmoid(avg_out)
