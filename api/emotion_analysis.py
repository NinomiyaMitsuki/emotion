from gensim.models import KeyedVectors
import os

# 使用gensim加载预训练中文分词embedding
# cn_model = KeyedVectors.load_word2vec_format("Chinese-Word-Vectors-master/sgns.zhihu.char",binary=False)

# embedding_dim = cn_model['山东大学'].shape[0]
#print('词向量的长度为{}'.format(embedding_dim))
#print(cn_model['山东大学'])
# 计算相似度
#print(cn_model.similarity('土豆', '马铃薯'))

pos_txts = os.listdir('pos')
neg_txts = os.listdir('neg')

print( '样本总共: '+ str(len(pos_txts) + len(neg_txts)))

# 现在我们将所有的评价内容放置到一个list里

train_texts_orig = [] # 存储所有评价，每例评价为一条string

# 添加完所有样本之后，train_texts_orig为一个含有4000条文本的list
# 其中前2000条文本为正面评价，后2000条为负面评价

for i in range(len(pos_txts)):
    with open('pos/'+pos_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()
for i in range(len(neg_txts)):
    with open('neg/'+neg_txts[i], 'r', errors='ignore') as f:
        text = f.read().strip()
        train_texts_orig.append(text)
        f.close()


print(len(train_texts_orig))

