import argparse, yaml, os, shutil
import sed_eval
import pdb

def read_meta_yaml(filename):
    with open(filename, 'r') as infile:
        data = yaml.load(infile)
    return data

def prepare_result_txt(yaml_dir, result_dir, s):
    reference_dir=os.path.join(result_dir, 'reference_txt')
    estimate_dir=os.path.join(result_dir, 'estimate_txt')
    if os.path.exists(reference_dir):
        shutil.rmtree(reference_dir)
    os.makedirs(reference_dir)
    file_list = []
    for yaml_file in sorted(os.listdir(yaml_dir)):
        if yaml_file.endswith("yaml"):
            data=read_meta_yaml(os.path.join(yaml_dir,yaml_file))
        else:
            continue
        for item in data:
            base_name=os.path.splitext(item['mixture_audio_filename'])[0]
            class_name = base_name.split('_')[1]
	    if item['event_present']:
                reference_name=base_name+('_'+str(item['ebr'])+'_'+item['event_class'])
	        f=open(os.path.join(reference_dir,reference_name+'_reference.txt'), 'wt')
	        f.write(str(item['event_start_in_mixture_seconds'])+'\t'+
			str(item['event_end_in_mixture_seconds'])+'\t'+
			str(item['event_class'])
		       )
	        f.close()
	    else:
	        reference_name=base_name
	        f=open(os.path.join(reference_dir,reference_name+'_reference.txt'), 'wt')
	        f.close()
            if (str(item['ebr']) in s) or (class_name in s) or s=='all':
	        file_list.append({
				'reference_file': os.path.join(reference_dir, reference_name+'_reference.txt'),
				'estimated_file': os.path.join(estimate_dir, base_name+'.txt')
				})
    return file_list

def evaluate(event_labels, file_list):
    data = []
    # Get used event labels
    all_data = sed_eval.util.event_list.EventList()
    for file_pair in file_list:
        reference_event_list = sed_eval.io.load_event_list(file_pair['reference_file'])
        estimated_event_list = sed_eval.io.load_event_list(file_pair['estimated_file'])
        data.append({'reference_event_list': reference_event_list,
                     'estimated_event_list': estimated_event_list})
        all_data += reference_event_list
    #event_labels = all_data.unique_event_labels

    # Start evaluating

    # Create metrics classes, define parameters
    segment_based_metrics = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=event_labels,
                                                                 time_resolution=1)
    event_based_metrics = sed_eval.sound_event.EventBasedMetrics(event_label_list=event_labels,
							     evaluate_onset=True,
							     evaluate_offset=False,
                                                             t_collar=0.5)

    # Go through files
    for file_pair in data:
        segment_based_metrics.evaluate(file_pair['reference_event_list'],
                                   file_pair['estimated_event_list'])
        event_based_metrics.evaluate(file_pair['reference_event_list'],
                                 file_pair['estimated_event_list'])

    # Get only certain metrics
    overall_segment_based_metrics = segment_based_metrics.results_overall_metrics()
    print "Accuracy:", overall_segment_based_metrics['accuracy']['accuracy']

    # Or print all metrics as reports
    print segment_based_metrics
    print event_based_metrics
    overall=event_based_metrics.results_overall_metrics()
    class_wise=event_based_metrics.results_class_wise_metrics()

    result=[overall, class_wise]
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--param', type=str)

    args = parser.parse_args()
    event_labels=['babycry', 'glassbreak', 'gunshot']
    yaml_dir="data/mixture_data/devtest"
    result_dir="result"
    file_list=prepare_result_txt(yaml_dir, result_dir, args.param)
    result = evaluate(event_labels, file_list)
    print result[0]['error_rate']['error_rate']
