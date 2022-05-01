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

if __name__ == '__main__':
    import json
    vgg_init = 'experiment_results/vgg_initial/metrics.json'
    vgg_augment = 'experiment_results/vgg_with_augment/metrics.json'

    res_init = 'experiment_results/resnet18_initial/metrics.json'
    res_augment = 'experiment_results/resnet18_with_augmentation/metrics.json'

    with open(vgg_init, 'r') as f:
        vgg_init_data = json.load(f) 
    
    with open(res_init, 'r') as f:
        res_init_data = json.load(f) 

    with open(vgg_augment, 'r') as f:
        vgg_augment_data = json.load(f) 

    with open(res_augment, 'r') as f:
        res_augment_data = json.load(f) 

    plt.plot(vgg_init_data['train_loss'])
    plt.plot(vgg_init_data['val_loss'])
    plt.plot(vgg_init_data['train_acc'])
    plt.plot(vgg_init_data['val_acc'])
    plt.plot(vgg_init_data['train_precision'])
    plt.plot(vgg_init_data['val_precision'])
    plt.plot(vgg_init_data['train_recall'])
    plt.plot(vgg_init_data['val_recall'])
    plt.legend(['train_loss','val_loss','train_accuracy','val_accuracy','train_precision','val_precision','train_recall','val_recall'],loc='upper right')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.show()