import pickle
if __name__=='__main__':
    ws = pickle.load(open('WordSequence.pkl','rb'))
    print(type(ws))
    print(len(ws.dict))