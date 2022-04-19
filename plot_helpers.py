import matplotlib.pyplot as plt

def plot_learning_curve(train_data:list,val_data:list,title:str,output_path:str):
    fig,ax = plt.subplots()
    ax.set_title(title)
    ax.plot(train_data)
    ax.plot(val_data)
    ax.legend(['train','val'])
    ax.set_ylabel('Score')
    ax.set_xlabel('Epoch')
    plt.savefig(output_path)
    