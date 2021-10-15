python main.py --model=NN_MC --mode=train --task=regression
python main.py --model=NN_MC --mode=train --task=classification --n_epochs=1000
python main.py --model=NN_MC --mode=train --task=regression --adversarial_training
python main.py --model=NN_MC --mode=train --task=classification --n_epochs=1000 --adversarial_training


python main.py --model=Deep_Ensemble --mode=train --task=regression
python main.py --model=Deep_Ensemble --mode=train --task=classification --n_epochs=1000
python main.py --model=Deep_Ensemble --mode=train --task=regression --adversarial_training
python main.py --model=Deep_Ensemble --mode=train --task=classification --n_epochs=1000 --adversarial_training



python main.py --model=SWAG --task=regression --mode=train --n_epochs=2000
python main.py --model=SWAG --task=regression --mode=train --n_epochs=3000 --adversarial_training
python main.py --model=SWAG --task=classification --mode=train --n_epochs=2000
python main.py --model=SWAG --task=classification --mode=train --n_epochs=2000 --adversarial_training

python main.py --model=LSTM_MC --task=regression --mode=train --n_epochs=1000
python main.py --model=LSTM_MC --task=regression --mode=train --n_epochs=2000 --adversarial_training
python main.py --model=LSTM_MC --task=classification --mode=train --n_epochs=1000
python main.py --model=LSTM_MC --task=classification --mode=train --n_epochs=1000 --adversarial_training

python main.py --model=BNN --task=regression --mode=train --n_epochs=2500
python main.py --model=BNN --task=regression --mode=train --n_epochs=3000 --adversarial_training
python main.py --model=BNN --task=classification --mode=train --n_epochs=500
python main.py --model=BNN --task=classification --mode=train --n_epochs=500 --adversarial_training

python main.py --model=GNN_MC --task=regression --mode=train --n_epochs=1000
python main.py --model=GNN_MC --task=regression --mode=train --n_epochs=1000 --adversarial_training
python main.py --model=GNN_MC --task=classification --mode=train --n_epochs=1000
python main.py --model=GNN_MC--task=classification --mode=train --n_epochs=1000 --adversarial_training