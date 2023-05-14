import os 
import argparse
import numpy as np
import pytorch_lightning as pl
from matplotlib import pyplot as plt

from data import MovieLens_DataModule, Books_DataModule,Anime_DataModule
from models import MultiVAE
from models import MultiDAE
from metric import Recall_at_k_batch, NDCG_binary_at_k_batch
from trainer import Trainer


def cli_main(args):
    pl.seed_everything(args.seed)
    
    if(args.dataset == "book"):
        movielens_dm = Books_DataModule(args)
        if os.path.isdir('./data/book_crossing/pro_sg')==False:
            movielens_dm.setup()
        print("book")
    elif (args.dataset == "anime"):
        movielens_dm = Anime_DataModule(args)
        if os.path.isdir('./data/anime/pro_sg')==False:
            movielens_dm.setup()
        print("anime")

    else:    
        print("movie")
        movielens_dm = MovieLens_DataModule(args)
        if os.path.isdir('./data/ml-20m/pro_sg')==False:
            movielens_dm.setup()
    
     
    train_data = movielens_dm.load_data(stage="train")
    val_tr, val_te = movielens_dm.load_data(stage="validation")
    test_tr, test_te = movielens_dm.load_data(stage="test")
    n_items = movielens_dm.load_n_items()
    print("train_data")
    # print(train_data)
    print("val_tr")
    # print(val_tr)
    print("test_tr")
    # print(test_tr)
     
    p_dims = [ 100,200,600, n_items]
    trainer = Trainer(args, p_dims)
    epochs_arr = [i+1 for i in range(args.epochs)];
    best_ndcg = -np.inf
    
    train_loss_arr=[];val_loss_arr=[];val_n100=[]
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_data);
        
        
        val_loss, n100_list,n20_list,n10_list,r50_list, r20_list,r10_list,r5_list,r1_list = trainer.evaluate(
            val_tr, val_te, mode="validation")
        # train_loss_arr.append(train_loss);
        val_loss_arr.append(val_loss);
        val_n100.append(np.mean(n100_list));
        
        if(epoch%10==0):
            print(f"epoch={epoch} finished");

        result = np.mean(n100_list)
        if result > best_ndcg:
            trainer.save_model()
            best_ndcg = result
        # trainer.load_model()
        
        train_loss_arr.append(train_loss);
    plt.plot(epochs_arr, train_loss_arr, marker='o');
    plt.xlabel('Number of epochs')
    plt.ylabel('Train Loss')
    plt.title('Train Loss vs Epochs')
    plt.grid(True);
    plt.show();
    plt.plot(epochs_arr, val_loss_arr, marker='o');
    plt.xlabel('Number of epochs')
    plt.ylabel('validation loss')
    plt.title('Validation Loss vs Epochs')
    plt.grid(True);
    plt.show();
    plt.plot(epochs_arr, val_n100, marker='o');
    plt.xlabel('Number of epochs')
    plt.ylabel('Mean NPCG@100 loss')
    plt.title('Validation Set NPCG@100 vs Epochs')
    plt.grid(True);
    plt.show();
    
    test_loss, n100_list,n20_list,n10_list,r50_list, r20_list,r10_list,r5_list,r1_list  = trainer.evaluate(test_tr, test_te, mode="test")
    print("-"*89)
    print("[FINAL RESULT]")
    print("Test Loss= %.5f" % (test_loss))
    print("Test NDCG@100=%.5f (%.5f)" %
          (np.mean(n100_list), np.std(n100_list) / np.sqrt(len(n100_list))))
    print("Test NDCG@20=%.5f (%.5f)" %
          (np.mean(n20_list), np.std(n20_list) / np.sqrt(len(n20_list))))
    print("Test NDCG@10=%.5f (%.5f)" %
          (np.mean(n10_list), np.std(n10_list) / np.sqrt(len(n10_list))))

    print("-"*89);


    print("Test Recall@50=%.5f (%.5f)" %
        (np.mean(r50_list), np.std(r50_list) / np.sqrt(len(r50_list))))
    print("Test Recall@20=%.5f (%.5f)" %
        (np.mean(r20_list), np.std(r20_list) / np.sqrt(len(r20_list))))
    print("Test Recall@10=%.5f (%.5f)" %
        (np.mean(r10_list), np.std(r10_list) / np.sqrt(len(r10_list))))
    print("Test Recall@2=%.5f (%.5f)" %
        (np.mean(r5_list), np.std(r5_list) / np.sqrt(len(r5_list))))
    print("Test Recall@1=%.5f (%.5f)" %
        (np.mean(r1_list), np.std(r1_list) / np.sqrt(len(r1_list))))
    
    
    # epochs_arr = [i+1 for i in range(200)];
    
    # plt.plot()



if __name__ == "__main__":
    '''
    for BooksCrossing
    python main.py --dataset book --data_dir ./data/book_crossing --data_url https://github.com/caserec/Datasets-for-Recommender-Systems/raw/master/Processed%20Datasets/BookCrossing/book_crossing.zip
    
    '''
    parser = argparse.ArgumentParser(
        description="PyTorch Version Variational Autoencoders for Collaborative Filtering")
    parser.add_argument("--dataset", default="movie",
                        type=str, help="The input dataset")
    parser.add_argument("--data_dir", default="./data/ml-20m",
                        type=str, help="The input data dir")
    parser.add_argument("--data_url", default="https://files.grouplens.org/datasets/movielens/ml-20m.zip",
                        type=str, help="Download File URL")
    parser.add_argument("--ckpt_dir", default="./ckpt",
                        type=str, help="Path for saving model")
    parser.add_argument("--model_name", default="multi-vae", 
                        type=str, help="Model type selected in the [multi-vae, mutli-dae]:")
    
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                        help='weight decay coefficient')
    parser.add_argument("--batch_size", type=int, default=500,
                        help="batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="upper epoch limit")
    parser.add_argument("--total_anneal_steps", type=int, default=2000,
                        help="the total number of gradient updates for annealing")
    parser.add_argument("--anneal_cap", type=float, default=0.2,
                        help="largest annealing parameter")
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='report interval')
    
    parser.add_argument("--seed", type=int, default=98765,
                        help="random seed")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="num workers for dataloader")
    
     

    args = parser.parse_args()

    cli_main(args)
