# federated_learning_experiment
Federated_learning experiments on MNIST

Experiments of federated learning with varies of acclerators.<br>
The setting of the optimizer is flollowed below links:<br>
<br>
FedAvg: H Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, et al. Communication-efficient
learning of deep networks from decentralized data. arXiv preprint arXiv:1602.05629, 2016.<br>
<br>
FedProx: Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith.
Federated optimization in heterogeneous networks. arXiv preprint arXiv:1812.06127, 2018.<br>
<br>
Scaffold: Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and
Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In
International Conference on Machine Learning, pages 5132–5143. PMLR, 2020.<br>
<br>
requirement: pip install pytorch==1.4.0 ; pip install torchvision==0.5.0
<br>
run 'python3 main.py --mu 0.001 --optimizer FedProx --save_path niid_t_5_shuffle_0.5_prox/'
<br>
The loss/epoch is saved in the folder `/runs`, the test acc/epoch is saved in the folder `/test_acc`
