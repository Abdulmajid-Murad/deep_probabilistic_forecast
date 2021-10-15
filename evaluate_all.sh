python main.py --model=NN_MC --mode=evaluate --task=regression
python main.py --model=NN_MC --mode=evaluate --task=classification
python main.py --model=NN_MC --mode=evaluate --task=regression --adversarial_training
python main.py --model=NN_MC --mode=evaluate --task=classification --adversarial_training

python main.py --model=Deep_Ensemble --mode=evaluate --task=regression
python main.py --model=Deep_Ensemble --mode=evaluate --task=classification
python main.py --model=Deep_Ensemble --mode=evaluate --task=regression --adversarial_training
python main.py --model=Deep_Ensemble --mode=evaluate --task=classification --adversarial_training


python main.py --model=SWAG --task=regression --mode=evaluate
python main.py --model=SWAG --task=regression --mode=evaluate  --adversarial_training
python main.py --model=SWAG --task=classification --mode=evaluate
python main.py --model=SWAG --task=classification --mode=evaluate  --adversarial_training


python main.py --model=LSTM_MC --task=regression --mode=evaluate 
python main.py --model=LSTM_MC --task=regression --mode=evaluate  --adversarial_training
python main.py --model=LSTM_MC --task=classification --mode=evaluate 
python main.py --model=LSTM_MC --task=classification --mode=evaluate --adversarial_training

python main.py --model=BNN --task=regression --mode=evaluate 
python main.py --model=BNN --task=regression --mode=evaluate  --adversarial_training
python main.py --model=BNN --task=classification --mode=evaluate 
python main.py --model=BNN --task=classification --mode=evaluate --adversarial_training


python main.py --model=GNN_MC --task=regression --mode=evaluate 
python main.py --model=LSTM_MC --task=regression --mode=evaluate  --adversarial_training
python main.py --model=LSTM_MC --task=classification --mode=evaluate 
python main.py --model=LSTM_MC --task=classification --mode=evaluate --adversarial_training