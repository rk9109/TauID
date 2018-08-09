# Variables: Higgs -> TauTau
# SIGNAL_FILE="/data/t3home000/rinik/GenNtuple_parentage_GluGluHToTauTau_M125_13TeV_powheg_pythia8.z" 
# BACKGROUND_FILE=" /data/t3home000/rinik/GenNtuple_parentage_QCD_HT300to500_TuneCP5_13TeV-madgraph-pythia8.z"

# Variables: ZPrime -> TauTau
# SIGNAL_FILE="/data/t3home000/rinik/GenNtuple_parentage_ZprimeToTauTau_M-3000_TuneCP5_13TeV-pythia8-tauola.z"
# BACKGROUND_FILE="/data/t3home000/rinik/GenNtuple_parentage_QCD_HT1500to2000_TuneCP5_13TeV-madgraph-pythia8.z"

# Variables: PF Data
SIGNAL_FILE="/data/t3home000/rinik/tauTuple_GluGluHToTauTau_PU140.z"
BACKGROUND_FILE="/data/t3home000/rinik/tauTuple_QCD_PU140.z"

# Define alias
shopt -s expand_aliases
alias train='/usr/local/bin/python2.7'

# Run training
echo "training dense..." >> out.log
train train.py -s $SIGNAL_FILE -b $BACKGROUND_FILE -c config_onelayer.yml && echo "onelayer complete" >> out.log
mv saved-data/onelayer_base.hdf5 saved-data/dense_base.hdf5
train train.py -l saved-data/dense_base.hdf5 -c config_twolayer.yml && echo "twolayer complete" >> out.log
train train.py -l saved-data/dense_base.hdf5 -c config_threelayer.yml && echo "threelayer complete" >> out.log

echo "training dense_norm..." >> out.log
train train.py -s $SIGNAL_FILE -b $BACKGROUND_FILE -c config_onelayer_norm.yml && echo "onelayer_norm complete" >> out.log
mv saved-data/onelayer_norm_base.hdf5 saved-data/dense_norm_base.hdf5
train train.py -l saved-data/dense_norm_base.hdf5 -c config_twolayer_norm.yml && echo "twolayer_norm complete" >> out.log
train train.py -l saved-data/dense_norm_base.hdf5 -c config_threelayer_norm.yml && echo "threelayer_norm complete" >> out.log

echo "training sequence..." >> out.log
train train.py -s $SIGNAL_FILE -b $BACKGROUND_FILE -c config_lstm.yml && echo "lstm complete" >> out.log
mv saved-data/lstm_base.hdf5 saved-data/sequence_base.hdf5
train train.py -l saved-data/sequence_base.hdf5 -c config_gru.yml && echo "gru complete" >> out.log
train train.py -l saved-data/sequence_base.hdf5 -c config_conv1D.yml && echo "conv1D complete" >> out.log

echo "training dense..." >> out.log
train train.py -s $SIGNAL_FILE -b $BACKGROUND_FILE -c config_lstm_norm.yml && echo "lstm_norm complete" >> out.log
mv saved-data/lstm_norm_base.hdf5 saved-data/sequence_norm_base.hdf5
train train.py -l saved-data/sequence_norm_base.hdf5 -c config_gru.yml && echo "gru_norm complete" >> out.log
train train.py -l saved-data/sequence_norm_base.hdf5 -c config_conv1D.yml && echo "conv1D_norm complete" >> out.log

echo "training dense..." >> out.log
train train.py -s $SIGNAL_FILE -b $BACKGROUND_FILE -c config_conv2D.yml && echo "onelayer complete" >> out.log
mv saved-data/conv2D_base.hdf5 saved-data/image_base.hdf5
