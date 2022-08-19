import json
import os

data_dir = "./data"

def main():
    fold_dirs = os.listdir(data_dir)

    for fold_dir in fold_dirs:
        text = ""

        json_file = json.load(open('data/' + fold_dir + '/' + fold_dir + '.json','r',encoding="utf-8"))

        text = text + 'data/' + fold_dir + '/' + 'BireView.png'

        numbers = len(json_file['themes'])

        t = [0,0,0,0,0,0,0,0]

        for i in range(numbers):
            theme = json_file['themes'][i]['type']
            t[theme] = 1

        for i in t:
            text = text + " " + str(i)

        with open("./dataset.txt","a") as file:
            file.write(text + "\n")

    return

if __name__ == "__main__":
    main()