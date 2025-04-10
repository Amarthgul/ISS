


import pickle 

from .Misc import RectPath

savePath = r"resources/Saves/"


def Save(object, fileName):
    with open(RectPath(savePath+fileName), 'wb') as f:
        pickle.dump(object, f)


def Load(fileName):
    with open(RectPath(savePath+fileName), 'rb') as f:
        data = pickle.load(f)
    return data 





def main():
    a = 22
    Save(a, "test")


if __name__ == "__main__":
    main()
