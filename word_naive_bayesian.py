import xml_reader


CATEGORY = ['RESTAURANT#GENERAL', 'SERVICE#GENERAL', 'FOOD#QUALITY', 'DRINKS#QUALITY', 'DRINKS#STYLE_OPTIONS',
           'FOOD#PRICES', 'DRINKS#PRICES', 'RESTAURANT#MISCELLANEOUS', 'FOOD#STYLE_OPTIONS', 'LOCATION#GENERAL',
           'RESTAURANT#PRICES', 'AMBIENCE#GENERAL']


text_data_lst, category_data_lst = xml_reader.load_data_for_task_1_2()

word_count_dict={}
word_count_dict_group = {
'NIL' : {},
'RESTAURANT#GENERAL' : {},
'SERVICE#GENERAL' : {},
'FOOD#QUALITY' : {}, 
'DRINKS#QUALITY' : {},
'DRINKS#STYLE_OPTIONS' : {},
'FOOD#PRICES' : {},
'DRINKS#PRICES' : {},
'RESTAURANT#MISCELLANEOUS' : {},
'FOOD#STYLE_OPTIONS' : {},
'LOCATION#GENERAL' : {}, 
'RESTAURANT#PRICES' : {},
'AMBIENCE#GENERAL' : {}
}

for text_data, category_data in zip(text_data_lst, category_data_lst):
	for word in text_data :
		num_category = 0
		if len(category_data) == 0 :
			num_category = 1
		else :
			num_category = len(category_data)
		
		if word_count_dict.get(word.lower()) == None :
			word_count_dict[word.lower()] = num_category 
		else :
			word_count_dict[word.lower()] += num_category

		if len(category_data) == 0 : 
			if word_count_dict_group['NIL'].get(word.lower()) == None : 
				word_count_dict_group['NIL'][word.lower()] = 1 
			else : 
				word_count_dict_group['NIL'][word.lower()] += 1
		else :
			for category in category_data :
				if word_count_dict_group[category].get(word.lower()) == None :
					word_count_dict_group[category][word.lower()] = 1
				else : 
					word_count_dict_group[category][word.lower()] += 1

for category in CATEGORY :
	with open('data/{}.data'.format(category)) as f :
		for line in f.readlines():
			for word in line.strip().split() :
				if word_count_dict.get(word.lower()) == None :
					word_count_dict[word.lower()] = 1 
				else :
					word_count_dict[word.lower()] += 1 

				if word_count_dict_group[category].get(word.lower()) == None :
					word_count_dict_group[category][word.lower()] = 1
				else : 
					word_count_dict_group[category][word.lower()] += 1

total_word = sum(word_count_dict.values())
total_word_group={}

for category in CATEGORY :
	total_word_group[category] = sum(word_count_dict_group[category].values())

reference_count = 0
reference_count_group = {
'NIL' : 0,
'RESTAURANT#GENERAL' : 0,
'SERVICE#GENERAL' : 0,
'FOOD#QUALITY' : 0, 
'DRINKS#QUALITY' : 0,
'DRINKS#STYLE_OPTIONS' : 0,
'FOOD#PRICES' : 0,
'DRINKS#PRICES' : 0,
'RESTAURANT#MISCELLANEOUS' : 0,
'FOOD#STYLE_OPTIONS' : 0,
'LOCATION#GENERAL' : 0, 
'RESTAURANT#PRICES' : 0,
'AMBIENCE#GENERAL' : 0
}
with open('stop_words.txt','r') as f:
	line = f.readline()
	for word in line.strip().split(',') :
		reference_count += word_count_dict[word]
		for category in ['NIL'] + CATEGORY :
			try :
				reference_count_group[category] += word_count_dict_group[category][word]
			except KeyError :
				pass
			



with open('word_naive_bayes_probability_normalized.txt', 'w') as f:
	f.write('{} {}\n'.format(len(word_count_dict.keys()), 13))
	for word in word_count_dict.keys() :
		f.write('{} '.format(word.lower()))
		for category in ['NIL'] + CATEGORY :
			if word_count_dict_group[category].get(word.lower()) == None :
				f.write('0 ')
			else :
				word_prob = word_count_dict_group[category][word.lower()] / word_count_dict[word.lower()]
				reference_prob = reference_count_group[category] / reference_count 
				f.write('{} '.format(word_prob / reference_prob))
		f.write('\n')
	
total_word_group={}


