import pandas as pd
import pkuseg




#以默认配置加载模型
seg = pkuseg.pkuseg()

  #  [IN] content = "我叫马化腾，我想学区块链,你说好不好啊，天青色等烟雨，而我在等你，月色被打捞器，晕开了结局"
	# 	dict = utils.read("./dict.txt")
	# 	pku = pkuseg.pkuseg(user_dict=dict)
	# 	res = pku.cut(content)
	# 	print(res)
  # [OUT] loading model
	# 	finish
	# 	['我', '叫', '马', '化', '腾', '，', '我', '想', '学', '区', '块', '链', ',', '你', '说', '好', '不', '好', '啊', '，', '天', '青', '色', '等', '烟', '雨', '，', '而', '我', '在', '等', '你', '，', '月', '色', '被', '打', '捞', '器', '，', '晕', '开', '了', '结', '局']




df = pd.read_csv('/home/yyang2/data/yyang2/Data/交大NLP资料/data_cut.csv',encoding='gbk')

test = df['Analysis'][1]

out = seg.cut(test)
df['Diagnosis_cut_'] = df['Diagnosis'].map(lambda x: seg.cut(str(x)))
df['Analysis_cut_'] = df['Analysis'].map(lambda x: seg.cut(str(x)))
df.to_csv('/home/yyang2/data/yyang2/Data/交大NLP资料/data_cut.csv')


