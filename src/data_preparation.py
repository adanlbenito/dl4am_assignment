import librosa
import numpy as np
import pandas as pd
import re
import IPython.display as ipd
import xml.etree.ElementTree as ET
import math
import os
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.rcParams["figure.figsize"]=20,10

class SMTGuitar:
    def __init__(self, root = data_root):
        # Dataset root folder
        self.root = root 
        self.root_dir = self.root + "IDMT-SMT-GUITAR/"
        # Token for identifying licks
        self.token = 'Lick'
        # File extension
        self.extension = {
            'audio': 'wav',
            'annotation': 'xml'
        }
        
        # Audio and annotation directories for dataset 2 (licks)
        self.d2_dir = {
            "audio": self.root_dir + "dataset2/audio/",
            "annotation": self.root_dir + "dataset2/annotation/"
        }
        # Tokens for extraction of information from file suffix
        self.suffix_tokens = {
            'excitation_styles': {
                'F': 'FS',
                'K': 'MU',
                'M': 'PK'
                },
            'expression_styles': {
                'DN': 'DN',
                'V': 'VI',
                'S': 'SL',
                'B': 'BE',
                'H': 'HA',
                'N': 'NO'
            }
        }
        # Excitation and expression styles for 
        self.styles = {
            'excitation': {
                'FS': 'finger_style',
                'MU': 'palm_muted',
                'PK': 'picked'
            }, 
            'expression': {
                'NO': 'normal',
                'BE': 'bending',
                'DN': 'dead_notes',
                'HA': 'harmonics',
                'SL': 'slide',
                'VI': 'vibrato'
            }
        }
        # Index of polyphonic licks
        self.polyphonic_licks = [7, 8, 9, 10, 12]
        
        # Dataframe to hold data from licks
        self.df = pd.DataFrame(columns = ['name', 'lick', 'mono', 'guitar', 'excitation', 'expression', 'position', 'fs', 'len', 'ms', 'samples'])
        
        # Dataframe to hold annotations
        self.annotations = pd.DataFrame(columns = ['name', 'onset', 'offset', 'excitation', 'expression'])
        
    def extract_file_names(self, dir, ext):
        # Get list of audio files 
        files = librosa.util.find_files(dir, ext=ext) 
        # Filter files corresponding to licks
        files = list(filter(lambda f: self.token in f, files))
        # Get file names
        file_names = list(map(lambda f: f.split('/')[-1], files))
        # Remove extension from file name
        file_names = list(map(lambda f: f.split('.')[0], file_names))
        
        return file_names, files
    
    def extract_audio(self, sr='None'): 
        # Get list of audio files & names
        file_names, files = self.extract_file_names(self.d2_dir["audio"], self.extension["audio"])
        # Extract IDs 
        ids = list(map(lambda f: int(re.search(self.token + "(.*?)_", f).group(1)), file_names))        # Extract guitar type 
        guitar_type = list(map(lambda f: re.search("^(.*?)_", f).group(1), file_names))
        # Extract style suffix
        styles = list(map(lambda f: re.search(self.token + "\d{1,}_(.*?)$", f).group(1), file_names))
        # Parse styles from suffix
        styles = list(map(self.parse_style_from_suffix, styles));

        # Extract audio and append information to dataframe
        for i, f in enumerate(files):
            x, fs = librosa.load(f, sr=None)
            self.df = self.df.append({
                'name' : file_names[i], 
                'lick': ids[i],
                'mono': ids[i] not in self.polyphonic_licks,
                'guitar': guitar_type[i],
                'excitation': styles[i]['excitation'],
                'expression': styles[i]['expression'],
                'position': styles[i]['position'],
                'fs': fs,
                'ms': 1000 * x.shape[0]/fs,
                'len': x.shape[0],
                'samples': x
            }, ignore_index=True, verify_integrity=True)
       
        # Make certain columns numeric
        numeric_col = ['lick', 'fs', 'ms', 'len']
        self.df = self.df.apply(lambda s: pd.to_numeric(s) if s.name in numeric_col else s, axis=1)

        return self.df
    
    def extract_annotations(self):

        # File names (from audio files)
        file_names = self.df['name'].tolist()

        # Get xml file names
        xml_file_names, xml_files = self.extract_file_names(self.d2_dir["annotation"], self.extension["annotation"])

        # Check for discrepancies between audio and file names
        names_not_in_annotation = list(set(file_names) - set(xml_file_names))
        names_not_in_audio = list(set(xml_file_names) - set(file_names))

        # Remap discrepancies
        # DISCLAIMER: this is very specific to the IDMT_SMT_DATASET
        audio_to_annot_dict = dict(zip(sum( list(
                        map(lambda b: 
                            list(
                                filter(lambda a: b in a , names_not_in_audio)
                            ), names_not_in_annotation
                        )
                    ), []), names_not_in_annotation))

        # 'Fix' xml file names for compatibility
        xml_file_names = list(map(lambda x: audio_to_annot_dict[x] if x in audio_to_annot_dict else x, xml_file_names))

        # Create dictionary between names and paths
        xml_file_name_dict = zip(xml_file_names, xml_files)

        # Extract annotations from xml files
        for name, path in xml_file_name_dict:
            tree = ET.parse(path)
            root = tree.getroot()
            for e in root.findall('./transcription/event'):
                self.annotations = self.annotations.append({
                        'name' : name, 
                        'onset': float(e.find('onsetSec').text),
                        'offset': float(e.find('offsetSec').text),
                        'excitation': e.find('excitationStyle').text,
                        'expression': e.find('expressionStyle').text
                }, ignore_index=True, verify_integrity=True)    

        return self.annotations
        
    def extract_transcript(self):
        # Get names from annotation dataframe
        names = self.annotations['name'].unique()

        # Number of different expression styles
        n_exp = len(self.styles['expression'])

        self.df['transcript'] = None
        self.df['num_exp'] = None

        # For each lick
        for n in names:
            # Extract corresponding annotations
            f_annot = self.annotations[self.annotations['name'] == n]
            # Get lick information for corresponding file
            audio_df = self.df[self.df['name'] == n]
            # Get original id (from dataframe)
            orig_idx = audio_df.index
            # Get sample frequency
            fs = int(audio_df['fs'])
            # Get file length
            length = int(audio_df['len'])

            # Initialise transcript ndarray to zero
                # rows: expression styles
            transcript = np.zeros((n_exp, length), dtype=int)
            # Initialise dictionary for number of annotated expressions
            exp_in_transcript = dict.fromkeys(self.styles['expression'], 0)
            # For each style
            for idx, exp in enumerate(self.styles['expression']):
                # Get corresponding annotations
                e_annot = f_annot[f_annot['expression'] == exp]
                # For each annotation
                for i, annot in e_annot.iterrows():
                    exp_in_transcript[exp] = exp_in_transcript[exp]+1
                    # Find time bounds
                    bounds = [int(fs*annot['onset']), int(math.ceil(fs*annot['offset']))]
                    # Update corresponding row in transcript
                    transcript[idx, bounds[0]:bounds[1]] = 1
                    
            # Update transcript in dataframe
            self.df['transcript'][orig_idx] = [transcript]
            self.df['num_exp'][orig_idx] = [exp_in_transcript]
            
        return self.df['transcript'], self.df['num_exp']
    
    def get_simple_transcript(self):
        exp_keys = list(self.styles['expression'].keys())
        no_idx = exp_keys.index('NO')
        bool_mask = np.ones((len(exp_keys),), bool)
        bool_mask[no_idx] = False
        
        self.df['transcript_simple'] = None
        self.df['num_exp_simple'] = None

        for idx, row in self.df.iterrows():
            st = np.apply_along_axis(lambda x: 1*np.logical_or.reduce(x), 0, row['transcript'][bool_mask, :])
            self.df['transcript_simple'][idx] = st
            exp_obj = {'NO': row['num_exp']['NO'], 'EXP': 0}
            for e in exp_keys:
                if e is not 'NO':
                    exp_obj['EXP'] = exp_obj['EXP'] + row['num_exp'][e]
            self.df['num_exp_simple'][idx] = exp_obj
            
        return self.df['transcript_simple'], self.df['num_exp_simple']

        
    # Filter dataframe based on input dictionary
    def filter_by(self, filt_dict, orginal_df=None):
        o_df = self.df if not orginal_df else orginal_df 
        filt_items = o_df
        for key, value in filt_dict.items():
            if key in o_df:
                if o_df[key].dtype == 'object':
                    v = value if isinstance(value , list) else [value]
                    filt_items = filt_items[filt_items[key].str.contains('|'.join(v))]
                else:
                    filt_items = filt_items.loc[o_df[key]==value]
        return filt_items

    # Parse expression and excitation styles and playing position from file name
    def parse_style_from_suffix(self, style_str):
        styles = {
            'suffix': style_str,
            'excitation': '',
            'expression': '',
            'position': 0
        }       
        style_arr = style_str.split('_')
                
        # Extract position
        if len(style_arr) > 1:
            lage = re.search("Lage(.*?)", style_arr[1]).group(1)
            styles['position'] = int(lage) if lage != '' else 1
        
        style_arr = style_arr[0]

        # Extract excitation style
        for key in self.suffix_tokens['excitation_styles']:
            if key in style_arr:
                styles['excitation'] = self.suffix_tokens['excitation_styles'][key]
                style_arr = style_arr.replace(key, '')
                break
        
        # Extract expression style
        expr_styles = list()
        for key in self.suffix_tokens['expression_styles'].keys():
            if key in style_arr:
                expr_styles.append(self.suffix_tokens['expression_styles'][key])
                style_arr = style_arr.replace(key, '')
        styles['expression'] = ','.join(expr_styles)     
        
        return styles
    
    # Return playing data for IPython.display.audio
    def play_data(self, idx=None, name=None):
        if idx is not None and idx in self.df.index:
            lick = self.df.iloc[idx]
        if name is not None:
            lick = self.df.filter_by({'name': name})
        return {'data': lick['samples'], 'rate': lick['fs']}
    
    # Get row by order (ignoring original index)
    def get_by_order (self, el, df=None):
        if df is None:
            df = self.df
        if(el < len(df.index)):
            return df.iloc[el]
        else:
            return None
        
    def save_data(self, name='smt_guitar.pkl', path=None):
        loc = './' if not path else path 
        file = loc+name
        smt_guitar.df.to_pickle(file)
        
    def load_data(self, name='smt_guitar.pkl', path=None):
        loc = './' if not path else path 
        file = loc+name
        print(file)
        if os.path.isfile(file):
            print('Is file')
            with open(file, "rb") as f:
                try:
                    self.df = pd.read_pickle(file)
                    print('Load pickle')
                    return True, self.df
                except Exception: 
                    print('Except!')
                    pass
        return False, self.df
                       
        
    def plot_annotations(self, idx=None, name=None, simple=False):
        if idx is not None and idx in self.df.index:
            lick = self.df.iloc[idx]
        if name is not None:
            lick = self.df.filter_by({'name': name})
            
        transcript_type = 'transcript_simple' if simple else 'transcript'
        exp_type = 'num_exp_simple' if simple else 'num_exp'

        colors = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.BASE_COLORS.keys())
        timeline = np.arange(0, lick['len'], 1)

        norm_samples = lick['samples']
        norm_samples/= np.max(np.abs(norm_samples),axis=0)

        s_samples = {'name': 'samples', 'x': timeline,
                     'y': norm_samples, 'color':'k', 'linewidth':1}
        colors.remove(s_samples['color'])

        signals.append(s_samples)

        for i, exp in enumerate(lick[exp_type].keys()):
            annot = {'name': exp, 'x': timeline,
                     'y': 2*lick[transcript_type][i]-1, 'color': colors[i], 'linewidth':2}
            signals.append(annot)

        fig, ax = plt.subplots()
        for signal in signals:
            ax.plot(signal['x'], signal['y'], 
                    color=signal['color'], 
                    linewidth=signal['linewidth'],
                    label=signal['name'])

        # Enable legend
        ax.legend()
        ax.set_title(lick['name'])
        plt.show()

