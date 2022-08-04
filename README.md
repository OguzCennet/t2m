
python train_wordConditioned.py -batch_size 1 -cpk jl2p -curriculum 1 -dataset KITMocap -early_stopping 1 -exp 1 -f_new 8 -feats_kind rifke -losses "['SmoothL1Loss']" -lr 0.001 -mask "[0]" -model Seq2SeqConditioned9 -modelKwargs "{'hidden_size':1024, 'use_tp':False, 's2v':'lstm'}" -num_epochs 1 -path2data ../dataset/kit-mocap -render_list subsets/render_list -s2v 1 -save_dir save/model/ -tb 1 -time 16 -transforms "['zNorm']" 


python train_wordConditioned.py -batch_size 8 -cpk jl2p -curriculum 1 -dataset KITMocap -early_stopping 1 -exp 1 -f_new 8 -feats_kind rifke -losses "['SmoothL1Loss']" -lr 0.001 -mask "[0]" -model Seq2SeqConditioned9 -modelKwargs "{'hidden_size':1024, 'use_tp':False, 's2v':'lstm'}" -num_epochs 300 -path2data ../dataset/kit-mocap -render_list subsets/render_list -s2v 1 -save_dir save/model/ -tb 1 -time 16 -transforms "['zNorm']" 

