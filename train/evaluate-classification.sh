# Variables
# [ include variables ]

# Run evaluation
echo "evaluating dense..." >> out.log
python evaluate.py -l saved-data/dense_base.hdf5 -d saved-models/onelayer_base/ -c config_onelayer.yml && \
echo "onelayer complete" >> out.log
python evaluate.py -l saved-data/dense_base.hdf5 -d saved-models/twolayer_base/ -c config_twolayer.yml && \
echo "twolayer complete" >> out.log
python evaluate.py -l saved-data/dense_base.hdf5 -d saved-models/threelayer_base/ -c config_threelayer.yml && \
echo "threelayer complete" >> out.log

echo "evaluating dense_norm..." >> out.log
python evaluate.py -l saved-data/dense_norm_base.hdf5 -d saved-models/onelayer_norm_base/ -c config_onelayer_norm.yml && \
echo "onelayer_norm complete" >> out.log
python evaluate.py -l saved-data/dense_norm_base.hdf5 -d saved-models/twolayer_norm_base/ -c config_twolayer_norm.yml && \
echo "twolayer_norm complete" >> out.log
python evaluate.py -l saved-data/dense_norm_base.hdf5 -d saved-models/threelayer_norm_base/ -c config_threelayer_norm.yml && \
echo "threelayer_norm complete" >> out.log

echo "evaluating sequence..." >> out.log
python evaluate.py -l saved-data/sequence_base.hdf5 -d saved-models/lstm_base/ -c config_lstm.yml && \
echo "lstm complete" >> out.log
python evaluate.py -l saved-data/sequence_base.hdf5 -d saved-models/gru_base/ -c config_gru.yml && \
echo "gru complete" >> out.log
python evaluate.py -l saved-data/sequence_base.hdf5 -d saved-models/conv1D_base/ -c config_conv1D.yml && \
echo "conv1D complete" >> out.log

echo "evaluating sequence_norm..." >> out.log
python evaluate.py -l saved-data/sequence_norm_base.hdf5 -d saved-models/lstm_norm_base/ -c config_lstm_norm.yml && \
echo "lstm_norm complete" >> out.log
python evaluate.py -l saved-data/sequence_norm_base.hdf5 -d saved-models/gru_norm_base/ -c config_gru_norm.yml && \
echo "gru_norm complete" >> out.log
python evaluate.py -l saved-data/sequence_norm_base.hdf5 -d saved-models/conv1D_norm_base/ -c config_conv1D_norm.yml && \
echo "conv1D_norm complete" >> out.log

echo "evaluating image..." >> out.log
python evaluate.py -l saved-data/image_base.hdf5 -d saved-models/conv2D_base/ -c config_conv2D.yml && \
echo "conv2D_norm complete" >> out.log

