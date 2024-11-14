# Deep Learning Theory

## About the Course
In recent years, deep learning has made great progress. It is widely used in many fields, such as computer vision, speech recognition, natural language processing, bioinformatics, etc., and also has a significant impact on basic science fields, such as applied and computational mathematics, statistical physics, computational chemistry and materials science, life science, etc. However, it should be noted that deep learning as a black-box model, its ability is explored through a large number of experiments. The theory of deep learning has gradually attracted the attention of many researchers, and has made progress in many aspects.

This course is closely around the latest development of deep learning theory. It intends to teach mathematical models, theories, algorithms and numerical experiments related to a series of basic problems from multiple perspectives of deep learning theory. This course is designed for doctoral, postgraduate and senior undergraduate students in all majors, who have basic knowledge of machine learning and deep learning.



The topics and the corresponding material are as follows:
  1. **Introduction to Deep Learning**  [material](#Introduction-to-deep-learning) [slides](./course_files/Lecture1.Overviewofdeeplearning.pdf)
  2. **Algorithmic Regularization** [material](#Algorithmic-Regularization) [slides](./course_files/Lecture2.AlgorithmicRegularization.pdf)
  3. **Inductive Biases due to Dropout** [material](#Inductive-Biases-due-to-Dropout) [slides](./course_files/Lecture3.InductiveBiasesduetoDropout.pdf)
  4. **Tractable Landscapes for Nonconvex Optimization** [material](#Tractable-Landscapes-for-Nonconvex-Optimization) [slides](./course_files/Lecture4.TractableLandscapes.pdf)
  5. **From Sparse Coding to Deep Learning** [material](#From-Sparse-Coding-to-Deep-Learning) [slides](./course_files/Lecture5.FromSparseCodingtoDeepLearning.pdf)
  6. **To be updated...**

[//]: # (  6. **Vulnerability of Deep Neural Networks** [material]&#40;#Vulnerability-of-Deep-Neural-Networks&#41; [slides]&#40;./course_files/Lecture6.VulnerabilityofDeepNeuralNetworks.pdf&#41;)

[//]: # (  7. **Information Bottleneck Theory** [material]&#40;#Information-Bottleneck-Theory&#41; [slides]&#40;./course_files/Lecture7.InformationBottleneckTheoryofDNNs.pdf&#41;)

[//]: # (  8. **Neural Tangent Kernel** [material]&#40;#Neural-Tangent-Kernel&#41; [slides]&#40;./course_files/Lecture8.NeuralTangentKernel.pdf&#41;)

[//]: # (  9. **Dynamic System and Deep Learning** [material]&#40;#Dynamic-System-and-Deep-Learning&#41; [slides]&#40;./course_files/Lecture9.DynamicSystemandDeepLearning.pdf&#41;)

[//]: # (  10. **Dynamic View of Deep Learning** [material]&#40;#Dynamic-View-of-Deep-Learning&#41; [slides]&#40;./course_files/Lecture10.DynamicViewofDeepLearning.pdf&#41;)

[//]: # (  11. **Generative Model** [material]&#40;#Generative-Model&#41; [slides1]&#40;./course_files/Lecture11.GenerativeModels-I.pdf&#41; [slides2]&#40;./course_files/Lecture12.GenerativeModels-II.pdf&#41;)

##  Prerequisites

Mathematical Analysis, Linear Algebra, Mathematical Statistics, Numerical Optimization, Matrix Theory, Fundamentals of Machine Learning and Deep Learning

## Previous Versions
The previous version of this course was taught in 2021. You can find it on [Deep Learning Theory (2021)](https://github.com/zhanglabtools/DeepLearningTheory.course).

## Introduction to Deep Learning

### Key papers
+ Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press.
+ Hornik, K., Stinchcombe, M., & White, H. (1989). Multilayer feedforward networks are universal approximators. Neural networks, 2(5), 359-366.
+ Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747.
+ Kramer, M. A. (1991). Nonlinear principal component analysis using autoassociative neural networks. AIChE journal, 37(2), 233-243.
+ Ledig, C., Theis, L., Huszár, F., Caballero, J., Cunningham, A., Acosta, A., ... & Shi, W. (2017). Photo-realistic single image super-resolution using a generative adversarial network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 4681-4690).
+ Lawrence, S., Giles, C. L., Tsoi, A. C., & Back, A. D. (1997). Face recognition: A convolutional neural-network approach. IEEE transactions on neural networks, 8(1), 98-113.
+ He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
+ Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).


## Algorithmic Regularization

### Key papers
+ Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo. (2019). Implicit regularization in deep matrix factorization, Advances in Neural Information Processing Systems 32 , 7413–7424.
+ Sanjeev Arora, Zhiyuan Li, and Abhishek Panigrahi. (2022). Understanding gradient descent on the edge of stability in deep learning, International Conference on Machine Learning, PMLR, pp. 948–1024.
+ Jonathon Byrd and Zachary Lipton.(2019)  What is the effect of importance weighting in deep learning? International Conference on Machine Learning, PMLR, pp. 872–881.
+ Jeremy Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. (2021). Gradient descent on neural networks typically occurs at the edge of stability, International Conference on Learning Representations.
+ Suriya Gunasekar, Jason Lee, Daniel Soudry, and Nathan Srebro. (2018). Characterizing implicit bias in terms of optimization geometry, International Conference on Machine Learning, PMLR, 2018, pp. 1832–1841.
+ Suriya Gunasekar, Blake Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur and Nathan Srebro. (2018). Implicit regularization in matrix factorization, 2018 Information Theory and Applications Workshop (ITA), IEEE, pp. 1–10.
+ Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. (2018). The implicit bias of gradient descent on separable data. The Journal of Machine Learning Research 19, no. 1, 2822–2878.
+ Lei Wu and Weijie J Su. (2023). The implicit regularization of dynamical stability in stochastic gradient descent, International Conference on Machine Learning, PMLR, pp. 37656–37684.
+ Da Xu, Yuting Ye, and Chuanwei Ruan. (2021). Understanding the role of importance weighting for deep learning, International Conference on Learning Representations.

## Inductive Biases due to Dropout

### Key papers
+ Baldi, P. and Hornik, K. (1989). Neural networks and principal component analysis: Learning from examples without local minima. Neural networks, 2(1):53–58.
+ Caruana, R., Lawrence, S., and Giles, L. (2001). Overfitting in neural nets: Backpropagation, conjugate gradient, and early stopping. Advances in neural information processing systems, pages 402–408.
+ Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(56):1929–1958.
+ Arora, R., Bartlett, P., Mianjy, P., and Srebro, N. (2021). Dropout: Explicit forms and capacity control. In International Conference on Machine Learning, pages 351–361. PMLR.
+ Mianjy, P., Arora, R., and Vidal, R. (2018). On the implicit bias of dropout. In International Conference on Machine Learning, pages 3540–3548. PMLR.
+ Xu, Z.-Q. J., Zhang, Y., and Luo, T. (2024). Overview frequency principle/spectral bias in deep learning. Communications on Applied Mathematics and Computation, pages 1–38.
+ Zhou, H., Qixuan, Z., Luo, T., Zhang, Y., and Xu, Z.-Q. (2022). Towards understanding the condensation of neural networks at initial training. Advances in Neural Information Processing Systems, 35:2184–2196.

## Tractable Landscapes for Nonconvex Optimization 

### Key papers

+ Ge, R., Lee, J. D., and Ma, T. (2016). Matrix completion has no spurious local minimum. In Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., and Garnett, R., editors, Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc.
+ Ge, R., Lee, J. D., and Ma, T. (2017). Learning one-hidden-layer neural networks with landscape design. arXiv preprint arXiv:1711.00501.
+ Garipov, T., Izmailov, P., Podoprikhin, D., Vetrov, D. P., and Wilson, A. G. (2018). Loss surfaces, mode connectivity, and fast ensembling of dnns. Advances in neural information processing systems, 31.
+ Frankle, J., Dziugaite, G. K., Roy, D., and Carbin, M. (2020). Linear mode connectivity and the lottery ticket hypothesis. In International Conference on Machine Learning, pages 3259–3269. PMLR.
+ Entezari, R., Sedghi, H., Saukh, O., and Neyshabur, B. (2021). The role of permutation invariance in linear mode connectivity of neural networks. arXiv preprint arXiv:2110.06296.
+ Ainsworth, S. K., Hayase, J., and Srinivasa, S. (2022). Git re-basin: Merging models modulo permutation symmetries. arXiv preprint arXiv:2209.04836.
+ Qu, X. and Horvath, S. (2024). Rethink model re-basin and the linear mode connectivity. arXiv preprint arXiv:2402.05966.

## From Sparse Coding to Deep Learning

### Key papers
+ Zhang, Z., & Zhang, S. (2021). Towards understanding residual and dilated dense neural networks via convolutional sparse coding. National Science Review, 8(3), nwaa159.
+ Papyan, V., Romano, Y., & Elad, M. (2017). Convolutional neural networks analyzed via convolutional sparse coding. The Journal of Machine Learning Research, 18(1), 2887-2938.
+ Papyan, V., Sulam, J., & Elad, M. (2017). Working locally thinking globally: Theoretical guarantees for convolutional sparse coding. IEEE Transactions on Signal Processing, 65(21), 5687-5701.
+ Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. Nature, 381(6583), 607-609.
+ Zeiler, M., Krishnan, D., Taylor, G., & Fergus, R. (2011). Deconvolutional Networks for Feature Learning. In Comput. Vis. Pattern Recognit.(CVPR), 2010 IEEE Conf (pp. 2528-2535).
+ He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
+ Pelt, D. M., & Sethian, J. A. (2018). A mixed-scale dense convolutional neural network for image analysis. Proceedings of the National Academy of Sciences, 115(2), 254-259.
+ Mingyang Li, Pengyuan Zhai, Shengbang Tong, Xingjian Gao, Shao-Lun Huang, Zhihui Zhu, Chong You, Yi Ma, et al. (2022). Revisiting sparse convolutional model for visual recognition, Advances in Neural Information Processing Systems 35. 10492–10504.

[//]: # (## Vulnerability of Deep Neural Networks)

[//]: # ()
[//]: # (### Key papers)

[//]: # (+ Fawzi, A., Fawzi, H., & Fawzi, O. &#40;2018&#41;. Adversarial vulnerability for any classifier. arXiv preprint arXiv:1802.08686.)

[//]: # (+ Shafahi, A., Huang, W. R., Studer, C., Feizi, S., & Goldstein, T. &#40;2018&#41;. Are adversarial examples inevitable?. arXiv preprint arXiv:1809.02104.)

[//]: # (+ Li, J., Ji, R., Liu, H., Liu, J., Zhong, B., Deng, C., & Tian, Q. &#40;2020&#41;. Projection & probability-driven black-box attack. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition &#40;pp. 362-371&#41;.)

[//]: # (+ Li, Y., Li, L., Wang, L., Zhang, T., & Gong, B. &#40;2019, May&#41;. Nattack: Learning the distributions of adversarial examples for an improved black-box attack on deep neural networks. In International Conference on Machine Learning &#40;pp. 3866-3876&#41;. PMLR.)

[//]: # (+ Wu, A., Han, Y., Zhang, Q., & Kuang, X. &#40;2019, July&#41;. Untargeted adversarial attack via expanding the semantic gap. In 2019 IEEE International Conference on Multimedia and Expo &#40;ICME&#41; &#40;pp. 514-519&#41;. IEEE.)

[//]: # (+ Rathore, P., Basak, A., Nistala, S. H., & Runkana, V. &#40;2020, July&#41;. Untargeted, Targeted and Universal Adversarial Attacks and Defenses on Time Series. In 2020 International Joint Conference on Neural Networks &#40;IJCNN&#41; &#40;pp. 1-8&#41;. IEEE.)

[//]: # ()
[//]: # (## Information Bottleneck Theory)

[//]: # ()
[//]: # (### Key papers)

[//]: # (+ Shwartz-Ziv, R., & Tishby, N. &#40;2017&#41;. Opening the black box of deep neural networks via information. arXiv preprint arXiv:1703.00810.)

[//]: # (+ Tishby, N., Pereira, F. C., & Bialek, W. &#40;2000&#41;. The information bottleneck method. arXiv preprint physics/0004057.)

[//]: # (+ Tishby, N., & Zaslavsky, N. &#40;2015, April&#41;. Deep learning and the information bottleneck principle. In 2015 IEEE Information Theory Workshop &#40;ITW&#41; &#40;pp. 1-5&#41;. IEEE.)

[//]: # (+ Saxe, A. M., Bansal, Y., Dapello, J., Advani, M., Kolchinsky, A., Tracey, B. D., & Cox, D. D. &#40;2019&#41;. On the information bottleneck theory of deep learning. Journal of Statistical Mechanics: Theory and Experiment, 2019&#40;12&#41;, 124020.)

[//]: # (+ Kolchinsky, A., Tracey, B. D., & Wolpert, D. H. &#40;2019&#41;. Nonlinear information bottleneck. Entropy, 21&#40;12&#41;, 1181.)

[//]: # (+ Achille, A., & Soatto, S. &#40;2018&#41;. Information dropout: Learning optimal representations through noisy computation. IEEE transactions on pattern analysis and machine intelligence, 40&#40;12&#41;, 2897-2905.)

[//]: # (+ Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. &#40;2016&#41;. Deep variational information bottleneck. arXiv preprint arXiv:1612.00410.)

[//]: # ()
[//]: # (## Neural Tangent Kernel)

[//]: # ()
[//]: # (### Key papers)

[//]: # (+ Jacot, A., Gabriel, F., & Hongler, C. &#40;2018&#41;. Neural tangent kernel: Convergence and generalization in neural networks. arXiv preprint arXiv:1806.07572.)

[//]: # (+ Lee, J., Xiao, L., Schoenholz, S., Bahri, Y., Novak, R., Sohl-Dickstein, J., & Pennington, J. &#40;2019&#41;. Wide neural networks of any depth evolve as linear models under gradient descent. Advances in neural information processing systems, 32, 8572-8583.)

[//]: # (+ Arora, S., Du, S., Hu, W., Li, Z., & Wang, R. &#40;2019, May&#41;. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In International Conference on Machine Learning &#40;pp. 322-332&#41;. PMLR.)

[//]: # (+ Arora, S., Du, S. S., Hu, W., Li, Z., Salakhutdinov, R., & Wang, R. &#40;2019&#41;. On exact computation with an infinitely wide neural net. arXiv preprint arXiv:1904.11955.)

[//]: # (+ Hu, W., Li, Z., & Yu, D. &#40;2019&#41;. Simple and effective regularization methods for training on noisily labeled data with generalization guarantee. arXiv preprint arXiv:1905.11368.)

[//]: # ()
[//]: # (## Dynamic System and Deep Learning)

[//]: # ()
[//]: # (### Key papers)

[//]: # (+ Weinan, E. &#40;2017&#41;. A proposal on machine learning via dynamical systems. Communications in Mathematics and Statistics, 5&#40;1&#41;, 1-11.)

[//]: # (+ Li, Q., Chen, L., & Tai, C. &#40;2017&#41;. Maximum principle based algorithms for deep learning. arXiv preprint arXiv:1710.09513.)

[//]: # (+ Parpas, P., & Muir, C. &#40;2019&#41;. Predict globally, correct locally: Parallel-in-time optimal control of neural networks. arXiv preprint arXiv:1902.02542.)

[//]: # (+ Haber, E., & Ruthotto, L. &#40;2017&#41;. Stable architectures for deep neural networks. Inverse problems, 34&#40;1&#41;, 014004.)

[//]: # (+ Lu, Y., Zhong, A., Li, Q., & Dong, B. &#40;2018, July&#41;. Beyond finite layer neural networks: Bridging deep architectures and numerical differential equations. In International Conference on Machine Learning &#40;pp. 3276-3285&#41;. PMLR.)

[//]: # (+ Li, Z., & Shi, Z. &#40;2017&#41;. Deep residual learning and pdes on manifold. arXiv preprint arXiv:1708.05115.)

[//]: # ()
[//]: # ()
[//]: # (## Dynamic View of Deep Learning)

[//]: # ()
[//]: # (### Key papers)

[//]: # (+ Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. &#40;2018&#41;. Neural ordinary differential equations. arXiv preprint arXiv:1806.07366.)

[//]: # (+ Yan, H., Du, J., Tan, V. Y., & Feng, J. &#40;2019&#41;. On robustness of neural ordinary differential equations. arXiv preprint arXiv:1910.05513.)

[//]: # (+ Gai, K., & Zhang, S. &#40;2021&#41;. A Mathematical Principle of Deep Learning: Learn the Geodesic Curve in the Wasserstein Space. arXiv preprint arXiv:2102.09235.)

[//]: # (+ Thorpe, M., & van Gennip, Y. &#40;2018&#41;. Deep limits of residual neural networks. arXiv preprint arXiv:1810.11741.)

[//]: # (+ Lu, Y., Ma, C., Lu, Y., Lu, J., & Ying, L. &#40;2020, November&#41;. A mean field analysis of deep ResNet and beyond: Towards provably optimization via overparameterization from depth. In International Conference on Machine Learning &#40;pp. 6426-6436&#41;. PMLR.)

[//]: # ()
[//]: # (## Generative Model)

[//]: # ()
[//]: # (### Key papers)

[//]: # (+ Kingma, D. P., & Welling, M. &#40;2013&#41;. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114.)

[//]: # (+ Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., ... & Bengio, Y. &#40;2014&#41;. Generative adversarial nets. Advances in neural information processing systems, 27.)

[//]: # (+ Arjovsky, M., & Bottou, L. &#40;2017&#41;. Towards principled methods for training generative adversarial networks. arXiv preprint arXiv:1701.04862.)

[//]: # (+ An, D., Guo, Y., Zhang, M., Qi, X., Lei, N., & Gu, X. &#40;2020, August&#41;. AE-OT-GAN: Training GANs from data specific latent distribution. In European Conference on Computer Vision &#40;pp. 548-564&#41;. Springer, Cham.)

[//]: # (+ Arjovsky, M., Chintala, S., & Bottou, L. &#40;2017, July&#41;. Wasserstein generative adversarial networks. In International conference on machine learning &#40;pp. 214-223&#41;. PMLR.)

[//]: # (+ Tolstikhin, I., Bousquet, O., Gelly, S., & Schoelkopf, B. &#40;2017&#41;. Wasserstein auto-encoders. arXiv preprint arXiv:1711.01558.)

[//]: # (+ Lei, N., An, D., Guo, Y., Su, K., Liu, S., Luo, Z., ... & Gu, X. &#40;2020&#41;. A geometric understanding of deep learning. Engineering, 6&#40;3&#41;, 361-374.)







