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

194 平均池化 189复现
output = self.sigmoid(avg_out)
0.17963
0.42602

195  TripletAttention
https://github.com/landskape-ai/triplet-attention/blob/master/MODELS/triplet_attention.py

196 平均池化 189复现 120epoch
output = self.sigmoid(avg_out)
0.1897
0.42092

197 https://github.com/GuHY777/MFENet-VIReID/blob/main/model/mfenet_no2.py
pool_result = torch.cat([avg_result, max_result], dim=1)
0.18935
0.38776

198 cbam sa
self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
0.15851
0.3801

199 197的基础上改进
pool_result = torch.cat([avg_result, max_result], dim=1)
self.conv1 = nn.Conv2d(2, 1, 7, padding=3, bias=False)
