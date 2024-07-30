import matplotlib.pyplot as plt


def plot_loss_and_acc(train_loss, val_loss, train_acc, val_acc, img_name):
    f, axs = plt.subplots(1, 2, figsize=(20, 5))
    f.suptitle('LOSS and ACC', size=12)
    axs[0].plot(train_loss, label='Training')
    axs[0].plot(val_loss, label='Validation')
    axs[0].set_title('Loss')
    axs[0].legend()
    axs[1].plot(train_acc, label='Training')
    axs[1].plot(val_acc, label='Validation')
    axs[1].set_title('Acc')
    plt.savefig("./image/"+img_name+".jpg")
    plt.close()
