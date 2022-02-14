# Create training dataset

Based on the Dockground database, this pipeline will create the input files to
train a neural network to predict interface residues.

## 1. Download the Dockground database (or create a database from scratch)
- Download the PDB files (abcd0A0B_1.pdb, abcd0A0B_2.pdb, bcde1A0B_1.pdb, ...) in a directory.
--> http://dockground.compbio.ku.edu/downloads/bound/templates/full_structures_v2.0.zip  
- Download the corresponding PDB ID list. (`list.txt` in the archive)

## 2. Filter the PDB ID list
- Remove the multimeric proteins or create a script to use them correctly. (`checklist.py`)
- Remove the proteins with non-standards amino acids. --> Can't encode them simply, so I removed them (no clue why, but I don't find the script I used to do this, but not hard to code)
- Remove the membrane proteins? (I didn't do it, but could be a good idea). List of membrane protein --> `rcsb_pdb_ids_20220212152248.txt` --> [link](https://www.rcsb.org/search?request=%7B%22query%22%3A%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22group%22%2C%22logical_operator%22%3A%22or%22%2C%22nodes%22%3A%5B%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_polymer_entity_annotation.type%22%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22PDBTM%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_polymer_entity_annotation.type%22%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22MemProtMD%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_polymer_entity_annotation.type%22%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22OPM%22%7D%7D%2C%7B%22type%22%3A%22terminal%22%2C%22service%22%3A%22text%22%2C%22parameters%22%3A%7B%22attribute%22%3A%22rcsb_polymer_entity_annotation.type%22%2C%22operator%22%3A%22exact_match%22%2C%22value%22%3A%22mpstruc%22%7D%7D%5D%7D%5D%2C%22logical_operator%22%3A%22and%22%2C%22label%22%3A%22text%22%7D%5D%2C%22logical_operator%22%3A%22and%22%7D%2C%22return_type%22%3A%22entry%22%2C%22request_info%22%3A%7B%22query_id%22%3A%22ed3380522c8737dad0e29990bb919d31%22%7D%2C%22request_options%22%3A%7B%22pager%22%3A%7B%22start%22%3A0%2C%22rows%22%3A25%7D%2C%22scoring_strategy%22%3A%22combined%22%2C%22sort%22%3A%5B%7B%22sort_by%22%3A%22score%22%2C%22direction%22%3A%22desc%22%7D%5D%7D%7D)
- Remove sequence identity. Create a multifasta file with the remaining sequences (`pdblist2Mfasta.py`)
--> Use CD-hit to remove seq identity. Something like 70% (neural network training) --> http://weizhong-lab.ucsd.edu/cdhit-web-server/cgi-bin/index.cgi
--> `00000000000.fas.1` --> PDB ID list (I don't have the script to go from multifasta to PDB ID list).
--> Use CD-hit to remove seq identity on `00000000000.fas.1` (70%) to create a 25% identity list (neural network evaluation). Filter only on interface residue would be best, but harder to implement.

**You now have two PDB ID list, one at 70% of homology (neural network training) and one at 25% homology (included in the 25% one).**

## 3. Create the PSSMs
- Create fasta files of the PDB. (`pdblist2fasta.py`)
- Use Psi-blast to create the PSSMs. I used the UniRef50 database (UniRef90 would be best, but longer to compute ~10 min/protein --> Paralelize the jobs on a cluster).

On the IFB, I used this script (not parallelized correctly)  

```
#!/bin/bash
#
#SBATCH -p long                      # partition
#SBATCH -N 1                         # nombre de nœuds
#SBATCH -n 12                        # nombre de cœurs
#SBATCH --mem 128GB                  # mémoire vive pour l'ensemble des cœurs
#SBATCH -t 15-0:00                   # durée maximum du travail (D-HH:MM)
#SBATCH -o slurm.%N.%j.out           # STDOUT
#SBATCH -e slurm.%N.%j.err           # STDERR

module add blast/2.9.0

FILES=~/FastaCport/*
for f in $FILES
do
  f="${f##*/}"
  f="${f%.fasta}"
  echo "$f"
  psiblast -query "/shared/home/yvandermeersche/FastaCport/$f.fasta" -db "/shared/bank/uniref50/current/blast/uniref50" -out "/shared/home/yvandermeersche/outCport/$f.txt" -out_ascii_pssm "/shared/home/yvandermeersche/pssmCport/$f.pssm" -num_iterations 2 -num_threads 12
done
```

## 4. Create intermediate data (used to create training / testing data)

- Create the ASA Nympy array (used to encode Y values). --> `get_interface_asa.py` with the previous  `list.txt` list.
- Encode X: PDB to Numpy array with all the features (len(seq) * nb_features) --> `encode_X.py`
- Encode window X: Use the Encode X file to convert each protein in w aa sliding window --> `window_X.py` --> len(seq) * window_size * nb_features

## 5. Create CV files
- Create the cross validation list (`create_CVlists.py`) --> Split the PDB ID list into an X folds cross validation. --> Mix the PDB ID so that every proteins is in the test set once.
- Create learn and val input files (`create_inputs.py`) with the 70% list
- Create test input files (`create_inputs_25%.py`) with the 25% list


## 6. Train neural network
- Train the network architecture (`do_cross_val.py`)  
**Train the network architecture with only one fold, then test it on the full CV.**
- Predict the test dataset to evaluate the performance of the model (`print_result_CV.py`)

## 7. Get the output of the tool for CV files
This one is a mess (`predict_CV_2.py`). I modified it a lot to do plots and stuff for my report, but it should output the same things as `main.py`.

