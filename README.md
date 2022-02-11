## 1. Dependencies should be installed using the following command before training:

   `pip install -r requirements.txt`


## 2. Data description:

   Publicly available Taxi and Limousine Commission (TLC) Taxi trip Data from

   https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

   We provide the pre-processed toy dataset of 84 time steps from 2019/4/09 0:00 a.m. to 2019/4/16 0:00 a.m. in ./data/ (2 hours per time step), including:

   #### ----train_data.npy
   
   The OD flows for training from Yellow Taxi records in sparse representation. The sparse OD matrix of each time step is a [Index, Value] list. Index = [(i_1,j_1),...,(i_k,j_k)], Value = [A_{i_1,j_1},...,A_{i_k,j_k}]. (70% of the whole existed OD pairs of Yellow Taxi)

   #### ----val_data.npy
   
   The OD flows of the last time steps for validation. (10% of the whole existed OD pairs of Yellow Taxi)

   #### ----test_data.npy
   
   The OD flows of the last time steps for testing. (20% of the whole existed OD pairs of Yellow Taxi)

   #### ----green_data.npy
   
   The OD flows for training from Green Taxi records.


## 3. Here are commands for training the model on NYC Taxi.

   `python train.py`


## 4. Evaluation.

   `python eval.py --checkpoints ./nyc/checkpoints`

## 5. Explainer.

   For current explanations: `python exp.py --task 0`
   
   For historical explanations: `python exp.py --task 1`

