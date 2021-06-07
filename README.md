# federated_learning_experiment
Federated_learning experiments on MNIST

Experiments of federated learning with varies of acclerators.<br>
The setting of the optimizer is flollowed below links:<br>

<br>
FedProx: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith.
Federated optimization in heterogeneous networks. arXiv preprint arXiv:1812.06127, 2018.<br>
<br>
Scaffold: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and
Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In
International Conference on Machine Learning, pages 5132–5143. PMLR, 2020.<br>
<br>
requirement: pip install pytorch==1.4.0 ; pip install torchvision==0.5.0<br>
<br>
For Scaffold optimizer run 'python3 run_main.py --optimizer scaffold --save_path niid_scaffold_lr0.001_simi0.0_400epoch --lr 0.001 --save_lossdir niid_scaffold_lr0.01_simi0.0_400epoch --save_accdir niid_scaffold_lr0.001_simi0.0_400epoch'<br>
<br>
For FedProx optimizer run 'python3 run_main.py --optimizer fedprox --save_path niid_scaffold_lr0.001_simi0.0_400epoch --lr 0.001 --save_lossdir niid_scaffold_lr0.01_simi0.0_400epoch --save_accdir niid_scaffold_lr0.001_simi0.0_400epoch'<br>
<br>
The loss/epoch is saved in the folder `/runs`, the test acc/epoch is saved in the folder `/test_acc`

![image](https://github.com/zhangxiaoman1995/federated_learning_experiment/blob/main/loss.png)
![image](https://github.com/zhangxiaoman1995/federated_learning_experiment/blob/main/acc.png)
