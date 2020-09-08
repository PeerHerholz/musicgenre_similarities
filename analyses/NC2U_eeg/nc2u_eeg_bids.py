# import necessary modules and functions
from os.path import join as opj
from nipype import Workflow, SelectFiles, Node, Function, DataSink, IdentityInterface, MapNode

# define important parameters of the experiment

# set task name
task_name = 'nc2u'

# set run break triggers
run_break_trigger = 210

# set bids root and source data directories
data_dir = '/Volumes/GameOfStora/NC2U_eeg/musicgenre_similarities/'
bids_root = '/Volumes/GameOfStora/NC2U_eeg/musicgenre_similarities/bids/'
source_data_path = '/Volumes/GameOfStora/NC2U_eeg/musicgenre_similarities/rawdata/'
working_dir = 'wdir_raw2bids'

# list of subject and identifiers
#subject_list = ['02', '03', '04', '05', '06', '07', '08', '09', '10']

subject_list = ['02']


run_list = ['1', '2', '3', '4', '5', '6', '7', '8']

# set event and label ids that define trigger-event mapping
event_id = {'alternative': 1, 'punk': 2, 'heavymetal': 3,
                'rocknroll': 4, 'psychedelic': 5, 'baroque': 6,
                'classic': 7, 'modernclassic': 8, 'renaissance': 9,
                'romantic': 10, 'deephouse': 11, 'drumandbass': 12,
                'dubstep': 13, 'techno': 14, 'trance': 15, 'funk': 16,
                'hiphop': 17, 'reggae': 18, 'rnb': 19, 'soul': 20, 'sound_off': 200,
                'run_break': 210
                }

labels = {'alternative': 0, 'punk': 1, 'heavymetal': 2,
          'rocknroll': 3, 'psychedelic': 4, 'baroque': 5,
          'classic': 6, 'modernclassic': 7, 'renaissance': 8,
          'romantic': 9, 'deephouse': 10, 'drumandbass': 11,
          'dubstep': 12, 'techno': 13, 'trance': 14, 'funk': 15,
          'hiphop': 16, 'reggae': 17, 'rnb': 18, 'soul': 19
          }


def split_runs(source_data_file, source_data_path, rb_trig):

    from os.path import join as opj
    import numpy as np
    import mne

    raw = mne.io.read_raw_bdf(source_data_file, preload=False)

    picks = mne.pick_types(raw.info, eeg=True)

    events = mne.find_events(raw)

    breaks = np.where(events[:, 2] == rb_trig)[0]

    run_files = []
    run_ids = []

    for i in range(0, len(breaks)):

        if i == 0:
            raw_tmp = raw.copy().crop(tmin=int(raw.times.min()),
                               tmax=int(events[:,0][breaks[0]]/1024))
        elif i == len(breaks):
            raw_tmp = raw.copy().crop(tmin=int(events[:,0][breaks[len(breaks)-1]]/1024),
                              tmax=int(raw.times.max()))
        else:
            raw_tmp = raw.copy().crop(tmin=int(events[:,0][breaks[i-1]]/1024),
                              tmax=int(events[:,0][breaks[i]]/1024))

        tmp_file = opj(source_data_path, source_data_file[source_data_file.rfind('/')+1:source_data_file.rfind('.')]
                       + '_run-%s' %str(i+1) + '.fif')

        run_files.append(tmp_file)
        run_ids.append(i+1)

        raw_tmp.save(tmp_file, overwrite=True)

    return(run_files, run_ids)


def raw2bids(source_data_run_file, bids_root, run_id, subject_id, task_name, event_id):

    import mne
    from mne_bids import make_bids_basename, write_raw_bids

    raw = mne.io.read_raw_fif(source_data_run_file, preload=False)

    picks = mne.pick_types(raw.info, eeg=True)

    events = mne.find_events(raw)

    # Recode genres that were sorted alphabetically to the desired integer assignments,
    # as noted in the event_id dict
    new_events = events[:, 2]

    for i in range(new_events.size):

        if (new_events[i] == 1):
            new_events[i] = 1
        elif (new_events[i] == 2):
            new_events[i] = 6
        elif (new_events[i] == 3):
            new_events[i] = 7
        elif (new_events[i] == 4):
            new_events[i] = 11
        elif (new_events[i] == 5):
            new_events[i] = 12
        elif (new_events[i] == 6):
            new_events[i] = 13
        elif (new_events[i] == 7):
            new_events[i] = 16
        elif (new_events[i] == 8):
            new_events[i] = 3
        elif (new_events[i] == 9):
            new_events[i] = 17
        elif (new_events[i] == 10):
            new_events[i] = 8
        elif (new_events[i] == 11):
            new_events[i] = 5
        elif (new_events[i] == 12):
            new_events[i] = 2
        elif (new_events[i] == 13):
            new_events[i] = 18
        elif (new_events[i] == 14):
            new_events[i] = 9
        elif (new_events[i] == 15):
            new_events[i] = 19
        elif (new_events[i] == 16):
            new_events[i] = 4
        elif (new_events[i] == 17):
            new_events[i] = 10
        elif (new_events[i] == 18):
            new_events[i] = 20
        elif (new_events[i] == 19):
            new_events[i] = 14
        elif (new_events[i] == 20):
            new_events[i] = 15

    events[:, 2] = new_events

    bids_basename = make_bids_basename(subject=subject_id,
                                       task=task_name, run=run_id)

    write_raw_bids(raw, bids_basename, bids_root, event_id=event_id,
                   events_data=events, overwrite=True)

# setup workflow nodes

# Create SelectFiles node
templates = {'eeg_raw': 'sub-{subject_id}.bdf'}

selectfiles = Node(SelectFiles(templates, base_directory= source_data_path),
          name='selectfiles')

# Create DataSink node
datasink = Node(DataSink(base_directory=data_dir,
                         container=bids_root),
                name="datasink")

# Infosource - a function free node to iterate over the list of subject names
infosource = Node(IdentityInterface(fields=['subject_id', 'run_id']),
                  name="infosource")
infosource.iterables = [('subject_id', subject_list),
                        ('run_id', run_list)]

# Create split run node
split_single_file = Node(Function(input_names=['source_data_file', 'source_data_path', 'rb_trig'],
                                  output_names=['run_files', 'run_ids'],
                                  function=split_runs),
                         name='split_single_file')
split_single_file.inputs.source_data_path = source_data_path
split_single_file.inputs.rb_trig = run_break_trigger

# Create raw_to_bids node
raw_to_bids = MapNode(Function(input_names=['source_data_run_file', 'bids_root', 'run_id', 'subject_id',
                                            'task_name', 'event_id'],
                          output_names=[],
                          function=raw2bids),
                      name='raw_to_bids', 
                      iterfield=['source_data_run_file', 'run_id'])
raw_to_bids.inputs.bids_root = bids_root
raw_to_bids.inputs.task_name = task_name
raw_to_bids.inputs.event_id = event_id

# Create a preprocessing workflow
raw2bids = Workflow(name='raw2bids')
raw2bids.base_dir = opj(data_dir, working_dir)

# Connect all components of the preprocessing workflow
raw2bids.connect([(infosource, selectfiles, [('subject_id', 'subject_id')]),
                  (infosource, raw_to_bids, [('subject_id', 'subject_id')]),
                  (selectfiles, split_single_file, [('eeg_raw', 'source_data_file')]),
                  (split_single_file, raw_to_bids, [('run_files', 'source_data_run_file')]),
                  (split_single_file, raw_to_bids, [('run_ids', 'run_id')]),
                 ])

raw2bids.run()