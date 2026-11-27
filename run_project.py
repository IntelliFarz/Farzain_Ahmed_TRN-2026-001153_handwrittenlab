"""Handwriting Lab experiments. Run from project root: python run_project.py --part all"""
import os, argparse, random, json, math, csv, warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'; PROC=ROOT/'processed_digits'; SHEETS=ROOT/'handwriting_sheets'; TABLES=ROOT/'tables'
for p in [OUT,PROC,TABLES]: p.mkdir(exist_ok=True)
S=1153
random.seed(S); np.random.seed(S)

def save_df(df,name):
    df.to_csv(OUT/(name+'.csv'),index=False)
    df.to_markdown(TABLES/(name+'.md'),index=False)


def load_handwriting():
    candidates = list(SHEETS.glob("*original.jpg")) + list(SHEETS.glob("*.jpg"))

    if not candidates:
        raise FileNotFoundError(
            "Put the original handwriting photo in handwriting_sheets/"
        )

    img = cv2.imread(str(candidates[0]))

    if img is None:
        raise ValueError("Could not read the handwriting image.")

    # Actual outer corners of the 5x10 grid in the supplied photograph
    src = np.float32([
        [218, 113],
        [1357, 141],
        [1362, 900],
        [232, 881]
    ])

    # Corrected grid dimensions: 10 columns x 5 rows
    dst = np.float32([
        [0, 0],
        [1000, 0],
        [1000, 700],
        [0, 700]
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, matrix, (1000, 700))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    X = []
    labels = []
    previews = []

    for r in range(5):
        for c in range(10):
            x1 = c * 100
            x2 = (c + 1) * 100
            y1 = r * 140
            y2 = (r + 1) * 140

            # Remove grid borders and keep the digit area
            cell = binary[y1 + 12:y2 - 12, x1 + 10:x2 - 10]

            # Remove small noise
            cell = cv2.medianBlur(cell, 3)

            # Find the handwritten digit
            points = cv2.findNonZero(cell)

            if points is None:
                canvas = np.zeros((8, 8), dtype=np.uint8)
            else:
                x, y0, w, h = cv2.boundingRect(points)

                digit = cell[
                    max(0, y0 - 3):min(cell.shape[0], y0 + h + 3),
                    max(0, x - 3):min(cell.shape[1], x + w + 3)
                ]

                dh, dw = digit.shape

                scale = 6.0 / max(dh, dw)
                new_w = max(1, int(dw * scale))
                new_h = max(1, int(dh * scale))

                resized = cv2.resize(
                    digit,
                    (new_w, new_h),
                    interpolation=cv2.INTER_AREA
                )

                canvas = np.zeros((8, 8), dtype=np.uint8)

                yy = (8 - new_h) // 2
                xx = (8 - new_w) // 2

                canvas[yy:yy + new_h, xx:xx + new_w] = resized

            # Convert to sklearn digits format (0-16)
            arr = np.rint(
                canvas.astype(np.float32) * 16 / 255
            ).clip(0, 16).astype(np.uint8)

            X.append(arr.flatten())
            labels.append(c)
            previews.append(arr)

    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)

    np.save(PROC / "handwriting_X_50.npy", X)
    np.save(PROC / "handwriting_y_50.npy", labels)

    fig, ax = plt.subplots(5, 10, figsize=(12, 7))

    for i, arr in enumerate(previews):
        ax.flat[i].imshow(
            arr,
            cmap="gray_r",
            vmin=0,
            vmax=16
        )
        ax.flat[i].axis("off")
        ax.flat[i].set_title(str(labels[i]), fontsize=9)

    fig.suptitle("Corrected Handwritten Digits (5x10)")
    fig.tight_layout()

    fig.savefig(
        OUT / "handwriting_5x10_grid.png",
        dpi=180
    )

    plt.close(fig)

    return X, labels, previews


def base_data():
    d=load_digits(); X=d.data.astype(np.float32); y=d.target.astype(np.int64)
    return train_test_split(X,y,test_size=.2,random_state=S,stratify=y)

def part_a():
    Xtr,Xte,ytr,yte=base_data(); Xhand,yhand,_=load_handwriting()
    models={'KNN':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=3)),
            'Logistic Regression':make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,random_state=S)),
            'Random Forest':RandomForestClassifier(n_estimators=300,random_state=S,n_jobs=-1)}
    rows=[]; best=None
    for name,m in models.items():
      m.fit(Xtr,ytr); p1=m.predict(Xte); p2=m.predict(Xhand)
      a1=accuracy_score(yte,p1); a2=accuracy_score(yhand,p2)
      rows.append({'Model':name,'sklearn_test_accuracy':a1,'handwriting_50_accuracy':a2})
      if best is None: best=(name,m,p2)
    df=pd.DataFrame(rows); save_df(df,'part_A_accuracy')
    cm=confusion_matrix(yhand,best[2],labels=list(range(10)))
    pd.DataFrame(cm,index=range(10),columns=range(10)).to_csv(OUT/'part_A_handwriting_confusion_matrix.csv')
    disp=ConfusionMatrixDisplay(cm,display_labels=range(10)); disp.plot(cmap='Blues'); plt.title('Handwriting confusion matrix - '+best[0]); plt.tight_layout(); plt.savefig(OUT/'part_A_confusion_matrix.png'); plt.close()
    # Side-by-side 10 sklearn images and handwritten grid
    d=load_digits(); fig,ax=plt.subplots(2,10,figsize=(12,3))
    for i in range(10): ax[0,i].imshow(d.images[i],cmap='gray_r'); ax[0,i].axis('off'); ax[0,i].set_title('sk '+str(d.target[i]))
    for i in range(10): ax[1,i].imshow(Xhand[i].reshape(8,8),cmap='gray_r',vmin=0,vmax=16); ax[1,i].axis('off'); ax[1,i].set_title('mine '+str(yhand[i]))
    fig.tight_layout(); fig.savefig(OUT/'part_A_comparison.png'); plt.close(fig)
    print('\nPART A\n',df.to_string(index=False),'\nBest-model handwriting confusion matrix saved.')
    return df

# NumPy network implementation, with stable softmax and explicit gradients.
def init_numpy(zero=False):
    rng=np.random.default_rng(S)
    W1=np.zeros((64,32)) if zero else rng.normal(0,np.sqrt(2/64),(64,32))
    b1=np.zeros(32); W2=np.zeros((32,10)) if zero else rng.normal(0,np.sqrt(2/32),(32,10)); b2=np.zeros(10)
    return [W1,b1,W2,b2]
def np_forward(X,params,relu=True):
    W1,b1,W2,b2=params; z1=X@W1+b1; h=np.maximum(0,z1) if relu else z1; z2=h@W2+b2; z2-=z2.max(axis=1,keepdims=True); ex=np.exp(z2); probs=ex/ex.sum(axis=1,keepdims=True); return probs,(X,z1,h,probs)
def np_loss(probs,y): return -np.log(np.clip(probs[np.arange(len(y)),y],1e-12,1)).mean()
def np_backward(y,params,cache,relu=True):
    X,z1,h,p=cache; n=len(y); dz2=p.copy(); dz2[np.arange(n),y]-=1; dz2/=n
    dW2=h.T@dz2; db2=dz2.sum(0); dh=dz2@params[2].T; dz1=dh*((z1>0) if relu else 1); dW1=X.T@dz1; db1=dz1.sum(0)
    return [dW1,db1,dW2,db2]
def part_b():
    Xtr,Xte,ytr,yte=base_data(); p=init_numpy(); lr=.08; losses=[]
    for ep in range(180):
      probs,cache=np_forward(Xtr,p); losses.append(np_loss(probs,ytr)); g=np_backward(ytr,p,cache)
      for a,b in zip(p,g): a-=lr*b
    pred=np_forward(Xte,p)[0].argmax(1); acc=accuracy_score(yte,pred)
    plt.figure(); plt.plot(losses); plt.xlabel('Epoch'); plt.ylabel('Training cross-entropy'); plt.title('NumPy NN training loss'); plt.tight_layout(); plt.savefig(OUT/'part_B_loss.png'); plt.close()
    # finite difference for 5 selected W2 values on small fixed batch
    xb=Xtr[:32]; yb=ytr[:32]; probs,cache=np_forward(xb,p); grads=np_backward(yb,p,cache); checks=[]; h=1e-5
    for i,j in [(0,0),(1,2),(5,3),(10,4),(20,8)]:
      old=p[2][i,j]; p[2][i,j]=old+h; lp=np_loss(np_forward(xb,p)[0],yb); p[2][i,j]=old-h; lm=np_loss(np_forward(xb,p)[0],yb); p[2][i,j]=old
      num=(lp-lm)/(2*h); ana=grads[2][i,j]; checks.append({'weight':f'W2[{i},{j}]','backprop':ana,'numerical':num,'absolute_difference':abs(ana-num)})
    save_df(pd.DataFrame(checks),'part_B_gradient_check')
    # required three experiments (brief, comparable fixed epochs)
    exp=[]
    for label,zero,relu,rate in [('all_zero',True,True,lr),('no_relu',False,False,lr),('100x_learning_rate',False,True,lr*100)]:
      q=init_numpy(zero=zero)
      for _ in range(20):
        pr,ca=np_forward(Xtr,q,relu=relu); gg=np_backward(ytr,q,ca,relu=relu)
        for a,b in zip(q,gg): a-=rate*b
      aa=accuracy_score(yte,np_forward(Xte,q,relu=relu)[0].argmax(1)); exp.append({'experiment':label,'test_accuracy':aa,'learning_rate':rate})
    save_df(pd.DataFrame(exp),'part_B_experiments')
    print(f'\nPART B: test accuracy={acc:.4f}; max gradient-check error={max(x["absolute_difference"] for x in checks):.3g}')
    return p,acc

# PyTorch models and experiments.
def part_c():
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    torch.manual_seed(S)
    Xtr,Xte,ytr,yte=base_data(); X_train,X_val,y_train,y_val=train_test_split(Xtr,ytr,test_size=.15,random_state=S,stratify=ytr)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    class M(nn.Module):
      def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,10))
      def forward(self,x): return self.net(x)
    results=[]; curves={}; best=None
    for lr in [0.0001,0.001,0.01,0.1,1,10]:
      torch.manual_seed(S); model=M().to(device); opt=torch.optim.SGD(model.parameters(),lr=lr); lossfn=nn.CrossEntropyLoss(); dl=DataLoader(TensorDataset(torch.tensor(X_train),torch.tensor(y_train)),batch_size=32,shuffle=True,generator=torch.Generator().manual_seed(S))
      curve=[]
      for ep in range(30):
        model.train(); total=0
        for xb,yb in dl:
          xb,yb=xb.to(device),yb.to(device); opt.zero_grad(); loss=lossfn(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(yb)
        curve.append(total/len(X_train))
      model.eval()
      with torch.no_grad():
        pv=model(torch.tensor(X_val,device=device)).argmax(1).cpu().numpy(); av=accuracy_score(y_val,pv)
      results.append({'learning_rate':lr,'validation_accuracy':av,'final_training_loss':curve[-1]}); curves[str(lr)]=curve
      if best is None or av>best[0]: best=(av,lr,model)
    df=pd.DataFrame(results); save_df(df,'part_C_learning_rates')
    plt.figure(figsize=(9,5))
    for k,v in curves.items(): plt.plot(v,label=k)
    plt.xlabel('Epoch'); plt.ylabel('Training loss'); plt.title('PyTorch learning-rate study'); plt.legend(title='Learning rate'); plt.tight_layout(); plt.savefig(OUT/'part_C_learning_rate_curves.png'); plt.close()
    torch.save(best[2].cpu().state_dict(),OUT/'part_C_best_model.pt')
    # load back and check identical predictions
    m2=M(); m2.load_state_dict(torch.load(OUT/'part_C_best_model.pt',map_location='cpu',weights_only=True)); m2.eval(); best[2].cpu().eval()
    xx=torch.tensor(Xte[:20]);
    with torch.no_grad(): same=torch.equal(best[2](xx).argmax(1),m2(xx).argmax(1))
    print('\nPART C\n',df.to_string(index=False),'\nReload predictions identical:',same)
    return df

def part_d():
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset,DataLoader
    torch.manual_seed(S); F=4+(S%5)
    Xtr,Xte,ytr,yte=base_data(); Xhand,yhand,_=load_handwriting()
    class CNN(nn.Module):
      def __init__(self,stride=1,drop=0.0):
        super().__init__(); self.conv1=nn.Conv2d(1,F,3,stride=stride,padding=1); self.pool=nn.MaxPool2d(2); self.conv2=nn.Conv2d(F,2*F,3,stride=1,padding=0); self.relu=nn.ReLU(); self.drop=nn.Dropout(drop); self.fc=nn.Linear(2*F*2*2,10)
      def forward(self,x): x=self.pool(self.relu(self.conv1(x))); x=self.relu(self.conv2(x)); x=x.flatten(1); return self.fc(self.drop(x))
    model=CNN(); params=sum(p.numel() for p in model.parameters())
    def nconv(n,k,s,p): return (n-k+2*p)//s+1
    # Architecture calculation, including pooling.
    n1=nconv(8,3,1,1); npool=n1//2; n2=nconv(npool,3,1,0)
    calc=(F*1*3*3+F)+(2*F*F*3*3+2*F)+(2*F*n2*n2*10+10)
    save_df(pd.DataFrame([{'F':F,'conv1_output':f'{F}x{n1}x{n1}','pool_output':f'{F}x{npool}x{npool}','conv2_output':f'{2*F}x{n2}x{n2}','flatten_features':2*F*n2*n2,'calculated_parameters':calc,'torch_parameters':params}]),'part_D_architecture')
    dl=DataLoader(TensorDataset(torch.tensor(Xtr.reshape(-1,1,8,8)/16,dtype=torch.float32),torch.tensor(ytr)),batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(S)); opt=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=1e-4); lossfn=nn.CrossEntropyLoss(); losses=[]
    for ep in range(35):
      model.train(); total=0
      for xb,yb in dl: opt.zero_grad(); l=lossfn(model(xb),yb); l.backward(); opt.step(); total+=l.item()*len(yb)
      losses.append(total/len(Xtr))
    model.eval()
    with torch.no_grad():
      ptest=model(torch.tensor(Xte.reshape(-1,1,8,8)/16,dtype=torch.float32)).argmax(1).numpy(); phand=model(torch.tensor(Xhand.reshape(-1,1,8,8)/16,dtype=torch.float32)).argmax(1).numpy()
    acc=accuracy_score(yte,ptest); hacc=accuracy_score(yhand,phand)
    torch.save(model.state_dict(),OUT/'part_D_cnn.pt')
    plt.figure(); plt.plot(losses); plt.xlabel('Epoch'); plt.ylabel('Training loss'); plt.title('CNN training loss'); plt.tight_layout(); plt.savefig(OUT/'part_D_cnn_loss.png'); plt.close()
    # learned first-layer filters and feature maps for first handwritten digit
    w=model.conv1.weight.detach().numpy(); fig,ax=plt.subplots(1,F,figsize=(2*F,2))
    for i in range(F): ax[i].imshow(w[i,0],cmap='coolwarm'); ax[i].axis('off'); ax[i].set_title(f'Filter {i}')
    fig.tight_layout(); fig.savefig(OUT/'part_D_filters.png'); plt.close()
    with torch.no_grad(): fmap=model.relu(model.conv1(torch.tensor(Xhand[:1].reshape(1,1,8,8)/16,dtype=torch.float32))).numpy()[0]
    fig,ax=plt.subplots(1,F,figsize=(2*F,2))
    for i in range(F): ax[i].imshow(fmap[i],cmap='viridis'); ax[i].axis('off'); ax[i].set_title(f'Map {i}')
    fig.tight_layout(); fig.savefig(OUT/'part_D_feature_maps.png'); plt.close()
    # overfit demonstration on only 100 samples with validation from remaining train set
    torch.manual_seed(S); small=CNN(drop=0); opt2=torch.optim.Adam(small.parameters(),lr=.01); criterion=nn.CrossEntropyLoss(); trloss=[]; valloss=[]
    xs=torch.tensor(Xtr[:100].reshape(-1,1,8,8)/16,dtype=torch.float32); ys=torch.tensor(ytr[:100]); xv=torch.tensor(Xtr[100:300].reshape(-1,1,8,8)/16,dtype=torch.float32); yv=torch.tensor(ytr[100:300])
    for ep in range(100):
      small.train(); opt2.zero_grad(); l=criterion(small(xs),ys); l.backward(); opt2.step(); trloss.append(l.item()); small.eval()
      with torch.no_grad(): valloss.append(criterion(small(xv),yv).item())
    plt.figure(); plt.plot(trloss,label='train'); plt.plot(valloss,label='validation'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Overfitting demonstration (100 training images)'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'part_D_overfitting.png'); plt.close()
    # stride-2 size prediction
    stride2_first=nconv(8,3,2,1); stride2_pool=stride2_first//2; stride2_second=nconv(stride2_pool,3,1,0)
    save_df(pd.DataFrame([{'test_set_accuracy':acc,'handwriting_50_accuracy':hacc,'stride2_conv1_size':stride2_first,'stride2_pool_size':stride2_pool,'stride2_conv2_size':stride2_second}]),'part_D_accuracy_stride')
    print(f'\nPART D: F={F}; parameters hand={calc}, torch={params}; sklearn test={acc:.4f}; handwriting={hacc:.4f}')
    return acc,hacc

def part_e():
    # Bug hunt based on supplied flawed script. One-at-a-time fixes are recorded conceptually.
    X,y=load_digits(return_X_y=True); X=X.astype(np.float32)
    # corrected: split before scaler, train-only fit; no softmax before CrossEntropyLoss;
    # zero gradients every batch; evaluate X_test not X_train; model.eval() for evaluation.
    Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=S,stratify=y)
    scaler=StandardScaler().fit(Xt); Xt=scaler.transform(Xt); Xv=scaler.transform(Xv)
    import torch
    from torch import nn
    torch.manual_seed(S)
    model=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.2),nn.Linear(32,10))
    opt=torch.optim.Adam(model.parameters(),lr=.003); lossfn=nn.CrossEntropyLoss(); tx=torch.tensor(Xt,dtype=torch.float32); ty=torch.tensor(yt,dtype=torch.long)
    for ep in range(120):
      model.train(); opt.zero_grad(); loss=lossfn(model(tx),ty); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad(): pred=model(torch.tensor(Xv,dtype=torch.float32)).argmax(1).numpy()
    acc=accuracy_score(yv,pred)
    bugs=[
      {'bug':1,'faulty_code':'StandardScaler().fit_transform(X) before split','correction':'Split first; fit scaler on training data only','concept':'Data leakage'},
      {'bug':2,'faulty_code':'Softmax(dim=1) before CrossEntropyLoss','correction':'Return raw logits to CrossEntropyLoss','concept':'Loss function expects logits'},
      {'bug':3,'faulty_code':'No opt.zero_grad() inside training loop','correction':'Clear gradients before each backward pass','concept':'Gradient accumulation'},
      {'bug':4,'faulty_code':'Evaluate predictions on X_train','correction':'Evaluate on held-out X_test','concept':'Train/test separation'},
      {'bug':5,'faulty_code':'No model.eval() for inference with Dropout','correction':'Call model.eval() before evaluation','concept':'Training vs inference mode'}]
    save_df(pd.DataFrame(bugs),'part_E_bug_table')
    (TABLES/'bug_table.md').write_text(pd.DataFrame(bugs).to_markdown(index=False)+'\n\nCorrected held-out accuracy: '+str(round(acc,4))+'\n',encoding='utf-8')
    print(f'\nPART E: corrected held-out accuracy={acc:.4f}; bug table saved.')
    return acc

def part_f1():
    digits=[int(c) for c in str(S)]; d=[0]+digits # pad to four digits (1153 -> already four)
    d=digits
    x1=np.array(d[:2],dtype=float)/5; x2=np.array(d[2:],dtype=float)/5; x3=np.array([1.,1.]); X=np.stack([x1,x2,x3])
    scores=X@X.T/np.sqrt(2); scores-=scores.max(axis=1,keepdims=True); weights=np.exp(scores); weights/=weights.sum(axis=1,keepdims=True); outputs=weights@X
    pd.DataFrame(weights,columns=['token1','token2','token3'],index=['query1','query2','query3']).to_csv(OUT/'part_F_attention_matrix.csv')
    pd.DataFrame(outputs,columns=['out_dim1','out_dim2']).to_csv(OUT/'part_F_attention_outputs.csv',index=False)
    print('\nPART F1: tokens=',X.tolist(),'\nAttention matrix:\n',weights,'\nToken 1 most attends to token',int(weights[0].argmax()+1))
    return weights

def part_f2():
    reviews=[
      ('The biryani was fragrant and perfectly cooked.', 'POSITIVE','English'),('Service was slow, but the food was excellent.','POSITIVE','English'),('The restaurant looked beautiful but the meal was disappointing.','NEGATIVE','English'),('Not bad at all; I would visit again.','POSITIVE','English'),('What a wonderful wait of ninety minutes.','NEGATIVE','English'),('The chai was warm and the staff were kind.','POSITIVE','English'),('The burger was dry and overpriced.','NEGATIVE','English'),('I liked the ambience, not the noisy music.','MIXED','English'),('Karachi street food was delicious tonight.','POSITIVE','English'),('The food was okay, nothing special.','NEUTRAL','English'),
      ('Biryani bohat zabardast thi, dobara aaonga.','POSITIVE','Roman Urdu'),('Service slow thi lekin taste acha tha.','MIXED','Roman Urdu'),('Khana bilkul acha nahi tha.','NEGATIVE','Roman Urdu'),('Chai achi thi aur staff bhi friendly tha.','POSITIVE','Roman Urdu'),('Wah, kya lajawab service hai, aik ghanta wait karwaya.','NEGATIVE','Roman Urdu'),('Ambience acha hai magar prices bohat high hain.','MIXED','Roman Urdu'),('Burger thanda tha, maza nahi aya.','NEGATIVE','Roman Urdu'),('Itna bura bhi nahi, theek tha.','NEUTRAL','Roman Urdu'),('Karachi ke rolls bohat tasty thay.','POSITIVE','Roman Urdu'),('Taste acha tha but safai bilkul nahi thi.','MIXED','Roman Urdu')]
    df=pd.DataFrame(reviews,columns=['sentence','your_label','language']); df.to_csv(TABLES/'sentence_table_input.csv',index=False)
    try:
      from transformers import pipeline
      clf=pipeline('sentiment-analysis'); outputs=clf(df.sentence.tolist())
      df['model_label']=[x['label'] for x in outputs]; df['model_score']=[x['score'] for x in outputs]
      # Standardize binary pipeline labels for accuracy only; neutral/mixed labels cannot be equated reliably.
      df['binary_reference']=df.your_label.map({'POSITIVE':'POSITIVE','NEGATIVE':'NEGATIVE'})
      df['binary_match']=np.where(df.binary_reference.notna(),df.binary_reference==df.model_label,np.nan)
      acc=df.groupby('language')['binary_match'].mean().to_dict()
      df.to_csv(TABLES/'sentence_table.csv',index=False); df.to_markdown(TABLES/'sentence_table.md',index=False)
      print('\nPART F2 results saved; binary accuracy (excluding mixed/neutral):',acc)
    except Exception as e:
      df['model_label']='NOT RUN'; df['model_score']=''; df.to_markdown(TABLES/'sentence_table.md',index=False)
      print('\nPART F2 not run: install transformers and torch, then rerun: python -m pip install transformers; model download requires internet. Error:',repr(e))
    return df

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--part',default='all',choices=['all','A','B','C','D','E','F1','F2']); args=ap.parse_args()
    print('HANDWRITING LAB | Farzain Ahmed | seed',S)
    if args.part in ('all','A'): part_a()
    if args.part in ('all','B'): part_b()
    if args.part in ('all','C'): part_c()
    if args.part in ('all','D'): part_d()
    if args.part in ('all','E'): part_e()
    if args.part in ('all','F1'): part_f1()
    if args.part in ('all','F2'): part_f2()
    print('\nFinished requested parts. See outputs/ and tables/.')
if __name__=='__main__': main()
