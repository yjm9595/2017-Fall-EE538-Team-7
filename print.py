import gensim
import csv


def filter_out_word(list,model):
    remove_list = []
    for K in list:
        if K not in model.vocab:
            remove_list.append(K)
    for remove_item in remove_list:
        list.remove(remove_item)
    print("update list: ")
    print(list)
    return list


model=gensim.models.KeyedVectors.load_word2vec_format('wiki.ko.vec')

test_list_1=['작가','아파트','소름','사건','광기','갓난아기','소설','이사','운전','택시','귀신','원혼']
test_list_2=['미아', '연주','재즈','향수','흑백','선율','연주자','작곡가','뮤지컬' ]
test_list_3=['아버지','캄보디아','캐릭터', '비밀결사','어드벤처','공중곡예']
test_list_4=['조직','보스','사투리','삼각관계','학교']
test_list_5=['한니발','악마','매혹','살인마','현상금','마약범','요원']

sentimental_keyword=['기쁨','사랑','분노','슬픔','달달한','몽환적인','웅장한','화려한','자극적인',
                     '빵빵터지는','짜증나는','오싹한','감동적인','짜증나는','훈훈', '우울','긴장감','스릴']


sentimental_keyword=filter_out_word(sentimental_keyword,model)
test_list_1=filter_out_word(test_list_1,model)
test_list_2=filter_out_word(test_list_2,model)
test_list_3=filter_out_word(test_list_3,model)
test_list_4=filter_out_word(test_list_4,model)
test_list_5=filter_out_word(test_list_5,model)





write_content=[]
for item in test_list_1:
    for s_keyword in sentimental_keyword:
        write_content.append([item,s_keyword,model.wv.similarity(item,s_keyword)])

for item in test_list_2:
    for s_keyword in sentimental_keyword:
        write_content.append([item,s_keyword,model.wv.similarity(item,s_keyword)])
for item in test_list_3:
    for s_keyword in sentimental_keyword:
        write_content.append([item,s_keyword,model.wv.similarity(item,s_keyword)])

for item in test_list_4:
    for s_keyword in sentimental_keyword:
        write_content.append([item,s_keyword,model.wv.similarity(item,s_keyword)])

for item in test_list_5:
    for s_keyword in sentimental_keyword:
        write_content.append([item,s_keyword,model.wv.similarity(item,s_keyword)])




with open('similarity_1.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow([ 'MovieKeyword','SentimentalKeyword','Similarity'])
    for item in write_content:
        MK=item[0]
        SK=item[1]
        similarity=item[2]
        writer.writerow([MK,SK,similarity])



