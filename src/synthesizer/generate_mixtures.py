import yaml, os, sys, shutil, librosa, soundfile as sf, numpy as np, pandas as pd
import argparse, textwrap
from IPython import embed
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from tqdm import tqdm
import hashlib
import warnings
import config as cfg
import pdb

def read_yaml(filename):
    try:
        with open(filename, 'r') as infile:
            data = yaml.load(infile)
    except IOError:
        print('Failed to open {}'.format(filename))
    return data


def write_yaml(filename, data):
    with open(filename, 'w') as outfile:
        outfile.write(yaml.dump(data,default_flow_style=False))


def list_audio_files(folder):
    files = []
    for dirpath,d,f in os.walk(folder):
        for file in f:
            if file[-4:].lower()=='.wav' or file[-5:].lower()=='.flac':
                files.append(os.path.join(dirpath,file))
    return files


def load_audio(path, target_fs=None):
    """
    Reads audio with (currently only one supported) backend (as opposed to librosa, supports more formats and 32 bit
    wavs) and resamples it if needed
    :param path: path to wav/flac/ogg etc (http://www.mega-nerd.com/libsndfile/#Features)
    :param target_fs: if None, original fs is kept, otherwise resampled
    :return:
    """
    y, fs = sf.read(path)
    if y.ndim>1:
        y = np.mean(y, axis=1)
    if target_fs is not None and fs!=target_fs:
        #print('Resampling %d->%d...' %(fs, target_fs))
        y = librosa.resample(y, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return y, fs


def write_audio(path, audio, fs, bitdepth=None):
    ext = os.path.splitext(path)[1].lower()
    if bitdepth==None:
        sf.write(file=path, data=audio, samplerate=fs)  # whatever is default in soundfile
    if bitdepth==24 and (ext=='.wav' or ext=='.flac'):
        #print('Writing 24 bits pcm, yay!')
        sf.write(file=path, data=audio, samplerate=fs, subtype='PCM_24')
    elif bitdepth==32:
        if ext=='.wav':
            sf.write(file=path, data=audio, samplerate=fs, subtype='PCM_32')
        else:
            print('Writing into {} format with bit depth {} is not supported, reverting to the default {}'.format(
                ext, bitdepth, sf.default_subtype(ext[1:])))
            sf.write(file=path, data=audio, samplerate=fs)
    elif bitdepth==16:
            sf.write(file=path, data=audio, samplerate=fs, subtype='PCM16')
    else:
        raise IOError('Unexpected bit depth {}'.format(bitdepth))


def pad_trunc_seq(x, max_len):
    """Pad or truncate a sequence data to a fixed length. 
    
    Args:
      x: ndarray, input sequence data. 
      max_len: integer, length of sequence to be padded or truncated. 
      
    Returns:
      ndarray, Padded or truncated input sequence data. 
    """
    L = len(x)
    shape = x.shape
    if L <= max_len:
        pad_shape = (max_len - L,) + shape[1:]
        pad = np.zeros(pad_shape)
        x_new = np.concatenate((x, pad), axis=0)
    else:
        start=np.random.randint(0,L-max_len)
        x_new = x[start:start+max_len]
    return x_new


def rmse(y):
    return np.sqrt(np.mean(np.abs(y) ** 2, axis=0, keepdims=False))


def get_event_amplitude_scaling_factor(signal, noise, target_ebr_db, method='rmse'):
    """
    Different lengths for signal and noise allowed: longer noise assumed to be stationary enough,
    and rmse is calculated over the whole signal
    """
    original_sn_rmse_ratio = rmse(signal) / rmse(noise)
    target_sn_rmse_ratio =  10 ** (target_ebr_db / float(20))
    signal_scaling_factor = target_sn_rmse_ratio/original_sn_rmse_ratio
    return signal_scaling_factor


def mix(bg_audio, event_audio, event_offset_samples, scaling_factor, magic_anticlipping_factor):
    """
    Mix np arrays of background and event audio (mono, non-matching lengths supported, sampling frequency better be the
    same, no operation in terms of seconds is performed though)
    :param bg_audio:
    :param event_audio:
    :param event_offset_samples:
    :param scaling_factor:
    :return:
    """
    old_event_audio = event_audio
    event_audio = scaling_factor*event_audio
    # check that the offset is not too long
    longest_possible_offset = len(bg_audio) - len(event_audio)
    if event_offset_samples > longest_possible_offset:
        raise AssertionError('Wrongly generated event offset: event tries to go outside the boundaries of the bg')
        #event_offset_samples = longest_possible_offset # shouldn't really happen if we pregenerate offset accounting for the audio lengths

    # measure how much to pad from the right
    tail_length = len(bg_audio) - len(event_audio) - event_offset_samples
    # pad zeros at the beginning of event signal
    padded_event = np.pad(event_audio, pad_width=((event_offset_samples, tail_length)), mode='constant', constant_values=0)
    if not len(padded_event)==len(bg_audio):
        raise AssertionError('Mixing yielded a signal of different length than bg! Should not happen')
    mixture = magic_anticlipping_factor* (padded_event + bg_audio)
    # Done! Now let's just confirm lengths mach
    # Also nice to make sure that we did not introduce clipping
    if np.max(np.abs(mixture)) >= 1:
        normalisation_factor = 1/float(np.max(np.abs(mixture)))
        print('Attention! Had to normalise the mixture by * %f' %normalisation_factor)
        print('I.e. bg max: %f, event max: %f, sum max: %f'
              %(np.max(np.abs(bg_audio)), np.max(np.abs(padded_event)), np.max(np.abs(mixture))))
        mixture /= np.max(np.abs(mixture))
        print('The scaling factor for the event was %f' %scaling_factor)
        print('The event before scaling was max %f' %np.max(np.abs(old_event_audio)))
    # now also refine the start time for the annotation
    return mixture #, start_time_seconds, end_time_seconds



def generate_mixture():
    m = hashlib.md5
    if os.path.exists(cfg.mixtures_dir):
        print('Folder {} already exists. Overwriting.'.format(cfg.mixtures_dir))
        shutil.rmtree(cfg.mixtures_dir)
    os.makedirs(cfg.mixtures_dir)

    r = np.random.RandomState(cfg.seed)
    #print('Current subset: %s' %subset)
    try:
        classwise_events = read_yaml(cfg.events_yaml_path)
        classwise_bgs = read_yaml(cfg.bgs_yaml_path)
    except IOError:
        sys.exit('Failed to load data, please set the function parameter of generate_mixture()---ratio(a two-dimensional float list).')

    allbgs = []
    for bg_name in classwise_bgs:
        allbgs += classwise_bgs[bg_name]
    for event_id,event_name in enumerate(classwise_events):
        print('Current class: {}'.format(event_name))
        cur_events = r.choice(classwise_events[event_name], int(round(cfg.mixtures_per_class*cfg.event_presence_prob)) )
        bgs = r.choice(allbgs, cfg.mixtures_per_class)

        event_presence_flags = (np.hstack((np.ones(len(cur_events)), np.zeros(len(bgs)-len(cur_events))))).astype(bool)
        event_presence_flags = r.permutation(event_presence_flags)
        event_instance_ids = np.nan*np.ones(len(bgs)).astype(int)  # by default event id set to nan: no event. fill it later with actual event ids when needed
        event_instance_ids[event_presence_flags] = np.arange(len(cur_events))

        target_ebrs = np.inf * np.ones(len(bgs))
        target_ebrs[event_presence_flags] = r.choice(cfg.ebrs, size=np.sum(event_presence_flags))

        mixture_recipes = []
        for i in tqdm(range(cfg.mixtures_per_class)):
            mixture_recipe = {}
            mixture_recipe['bg_path'] = str(bgs[i])
            mixture_recipe['event_present'] = bool(event_presence_flags[i])
            mixture_recipe['ebr'] = float(target_ebrs[i])
            mixture_hash = m(yaml.dump(mixture_recipe).encode('utf-8')).hexdigest()
            mixture_recipe['mixture_audio_filename'] = 'mixture' + '_' + event_name + '_' + str(target_ebrs[i]) + '_' + '%04d' % i + '_' + mixture_hash + '.wav'
            bg_path_full = os.path.join(cfg.bgs_dir, bgs[i])
            bg_audio, fs_bg = load_audio(bg_path_full, target_fs=cfg.common_fs)
            bg_audio=pad_trunc_seq(bg_audio, int(cfg.max_seconds*fs_bg))
            if(event_presence_flags[i]):
                assert not np.isnan(event_instance_ids[i])  # shouldn't happen, nans are in sync with falses in presence flags
                event_path_full=os.path.join(cfg.events_dir, cur_events[int(event_instance_ids[i])])
                event_audio,fs_event =load_audio(event_path_full, target_fs=cfg.common_fs)
                if not len(bg_audio) > len(event_audio):
                    print event_path_full
                    raise AssertionError("length of bg_audio > that of event_audio!")
                offset_sample = int((len(bg_audio)-len(event_audio))*r.rand())
                mixture_recipe['event_class'] = str(event_name)
                mixture_recipe['event_id'] = int(event_id+1)
                mixture_recipe['event_path'] = str(cur_events[int(event_instance_ids[i])])
                mixture_recipe['event_start_in_mixture_seconds'] = float(1.0*offset_sample/cfg.common_fs)
                mixture_recipe['event_end_in_mixture_seconds'] = float(1.0*(offset_sample+len(event_audio))/cfg.common_fs)
                mixture_recipe['annotation_string'] = mixture_recipe['mixture_audio_filename'] \
                                                          + '\t' + str(mixture_recipe['event_start_in_mixture_seconds']) \
                                                          + '\t' + str(mixture_recipe['event_end_in_mixture_seconds']) \

                eventful_part_of_bg = bg_audio[offset_sample:offset_sample+len(event_audio)]
                scaling_factor = get_event_amplitude_scaling_factor(event_audio, eventful_part_of_bg, target_ebrs[i]) 
                mixture = mix(bg_audio, event_audio, offset_sample, scaling_factor, cfg.magic_anticlipping_factor)
            else:
                mixture_recipe['annotation_string'] = mixture_recipe['mixture_audio_filename']
                mixture = cfg.magic_anticlipping_factor*bg_audio

            write_audio(os.path.join(cfg.mixtures_dir, mixture_recipe['mixture_audio_filename']), mixture, cfg.common_fs, cfg.bitdepth)

            mixture_recipes.append(mixture_recipe)

        if not os.path.exists(cfg.recipes_dir):
            os.makedirs(cfg.recipes_dir)
        write_yaml(os.path.join(cfg.recipes_dir, 'mixture_recipes_' + event_name + '.yaml'), mixture_recipes)
        #yaml.safe_dump(mixture_recipes)
        print('Mixture recipes dumped into file {} successfully'.format(os.path.join(cfg.recipes_dir, 'mixture_recipes_' + event_name + '.yaml')))
        print('-'*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--distribute', type=bool, default=False)

    args = parser.parse_args()
    if args.distribute:
        names=["bgs", "events"]
        ratio=[0.2, 0.2]
        yaml_dir="data/mixture_data"
        if os.path.exists(yaml_dir):
            shutil.rmtree(yaml_dir)
        os.makedirs(yaml_dir)
        for i,d in enumerate([cfg.bgs_dir, cfg.events_dir]):
            devtest={}
            devtrain={}
            for class_name in sorted(os.listdir(d)):
                if not devtest.has_key(class_name):
                    devtest[class_name]=[]
                if not devtrain.has_key(class_name):
                    devtrain[class_name]=[]
                audio_files=list_audio_files(os.path.join(d,class_name))
                np.random.shuffle(audio_files)
                num_of_devtest=int(ratio[i]/(ratio[i]+1)*len(audio_files))
                for j,audio_file in enumerate(audio_files):
                    temp=audio_file.split('/')
                    audio_filepath=os.path.join(temp[-2],temp[-1])
                    if(j<num_of_devtest):
                        devtest[class_name].append(audio_filepath)
                    else:
                        devtrain[class_name].append(audio_filepath)
            write_yaml(os.path.join(yaml_dir,names[i]+'_devtest.yaml'),devtest)
            write_yaml(os.path.join(yaml_dir,names[i]+'_devtrain.yaml'),devtrain)

    for phase in ["devtest", "devtrain"]:
        print('*'*80)
        print('Current subset: %s' %phase)
        print('*'*80)
        if phase=="devtrain":
            cfg.bgs_yaml_path = cfg.bgs_yaml_path.replace("devtest",phase)
            cfg.events_yaml_path = cfg.events_yaml_path.replace("devtest",phase)
            cfg.mixtures_dir = cfg.mixtures_dir.replace("devtest",phase)
            cfg.recipes_dir = cfg.recipes_dir.replace("devtest",phase)
            cfg.event_presence_prob = 1
            cfg.mixtures_per_class = 5000
        generate_mixture()
