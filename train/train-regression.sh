# Variables: Higgs -> TauTau
# SIGNAL_FILE="/data/t3home000/rinik/GenNtuple_parentage_GluGluHToTauTau_M125_13TeV_powheg_pythia8_regression.z" 
# TREE="GenNtupler/gentree"

# Variables: ZPrime -> TauTau
# SIGNAL_FILE="/data/t3home000/rinik/GenNtuple_parentage_ZprimeToTauTau_M-3000_TuneCP5_13TeV-pythia8-tauola.z"
# TREE="GenNtupler/gentree"

# Variables: PF Data
SIGNAL_FILE="/data/t3home000/rinik/tauTuple_GluGluHToTauTau_PU140_regression.z"
TREE="dumpP4/objects"

# Define alias
shopt -s expand_aliases
alias train='/usr/local/bin/python2.7'

# Run training
echo "training dense..." >> out.log
train train.py -s $SIGNAL_FILE -t $TREE -c config_onelayer.yml && echo "onelayer complete" >> out.log
mv saved-data/onelayer.hdf5 saved-data/dense.hdf5
mv saved-data/onelayer_scaler.pkl saved-data/dense_scaler.pkl
train train.py -l saved-data/dense.hdf5 -c config_twolayer.yml && echo "twolayer complete" >> out.log
train train.py -l saved-data/dense.hdf5 -c config_threelayer.yml && echo "threelayer complete" >> out.log

echo "training dense_norm..." >> out.log
train train.py -s $SIGNAL_FILE -t $TREE -c config_onelayer_norm.yml && echo "onelayer_norm complete" >> out.log
mv saved-data/onelayer_norm.hdf5 saved-data/dense_norm.hdf5
mv saved-data/onelayer_norm_scaler.hdf5 saved-data/dense_norm_scaler.hdf5
train train.py -l saved-data/dense_norm.hdf5 -c config_twolayer_norm.yml && echo "twolayer_norm complete" >> out.log
train train.py -l saved-data/dense_norm.hdf5 -c config_threelayer_norm.yml && echo "threelayer_norm complete" >> out.log

echo "training sequence..." >> out.log
train train.py -s $SIGNAL_FILE -t $TREE -c config_lstm.yml && echo "lstm complete" >> out.log
mv saved-data/lstm.hdf5 saved-data/sequence.hdf5
mv saved-data/lstm_scaler.hdf5 saved-data/sequence_scaler.hdf5
train train.py -l saved-data/sequence.hdf5 -c config_gru.yml && echo "gru complete" >> out.log
train train.py -l saved-data/sequence.hdf5 -c config_conv1D.yml && echo "conv1D complete" >> out.log

echo "training sequence_norm..." >> out.log
train train.py -s $SIGNAL_FILE -t $TREE -c config_lstm_norm.yml && echo "lstm_norm complete" >> out.log
mv saved-data/lstm_norm.hdf5 saved-data/sequence_norm.hdf5
mv saved-data/lstm_norm_scaler.hdf5 saved-data/sequence_norm_scaler.hdf5
train train.py -l saved-data/sequence_norm.hdf5 -c config_gru_norm.yml && echo "gru_norm complete" >> out.log
train train.py -l saved-data/sequence_norm.hdf5 -c config_conv1D_norm.yml && echo "conv1D_norm complete" >> out.log

echo "training image..." >> out.log
train train.py -s $SIGNAL_FILE -t $TREE -c config_conv2D.yml && echo "conv2D complete" >> out.log
mv saved-data/conv2D.hdf5 saved-data/image.hdf5
mv saved-data/conv2D_scaler.hdf5 saved-data/image_scaler.hdf5

