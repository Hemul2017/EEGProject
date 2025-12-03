

import csv
import matplotlib.pyplot as plt


def plot():
    with open('./results/train.csv', 'r') as train_csv:
        reader = csv.reader(train_csv)
        reader = list(reader)
        train_loss = [float(row[0]) for row in reader]
        train_acc = [float(row[1]) for row in reader]

    with open('./results/test.csv', 'r') as test_csv:
        reader = csv.reader(test_csv)
        reader = list(reader)
        test_loss = [float(row[0]) for row in reader]
        test_acc = [float(row[1]) for row in reader]


    plt.plot(train_loss, label='Train Loss')
    plt.plot(test_loss, label='Test Loss')
    plt.legend()
    plt.show()

    plt.plot(train_acc, label='Train Accuracy')
    plt.plot(test_acc, label='Test Accuracy')
    plt.legend()
    plt.show()



def main():
    plot()


if __name__ == '__main__':
    main()