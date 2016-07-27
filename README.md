# personidentification
Network to identify gait signatures in a small egocentric dataset with Keras implementation.


Images are read in and preprocessed with methods in imagereaders.py

Examples of preprocessed networks (ours and our benchmark) are in identifywearer.py and benchmark.py respectively.  

Our benchmark comes from Shmuel Peleg and Yedid Hosen, bibTeX citation below

@article{DBLP:journals/corr/HoshenP14,
  author    = {Yedid Hoshen and
               Shmuel Peleg},
  title     = {Egocentric Video Biometrics},
  journal   = {CoRR},
  volume    = {abs/1411.7591},
  year      = {2014},
  url       = {http://arxiv.org/abs/1411.7591},
  timestamp = {Mon, 01 Dec 2014 14:32:13 +0100},
  biburl    = {http://dblp.uni-trier.de/rec/bib/journals/corr/HoshenP14},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}

We use the "ball play" and "walk" actions from M.S. Ryoo's Dogcentric Dataset to test the networks (http://robotics.ait.kyushu-u.ac.jp/~yumi/db/first_dog.html) bibTeX citation below

@inproceedings{yumi2014first,
      title={First-Person Animal Activity Recognition from Egocentric Videos},
      author={Y. Iwashita and A. Takamine and R. Kurazume and M. S. Ryoo},
      booktitle={International Conference on Pattern Recognition (ICPR)},
      year={2014},
      month={August},
      address={Stockholm, Sweden},
} 



Networks are located in networks.py


Annotations are read in with a label, \t, # of occurrences.  Ex:

1 4

means there are 4 occurrences of object 1.  


When video is being preprocessed, a print statement will show how many clips are made from one video so annotations can be made easier.
