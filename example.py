#%%
import sys
sys.path.append('C:\\Users\\Subin Bang\\Dropbox\\EELS Data\\EELS_Filter')
from EELS import *


new = Spectrum()
new.set_dir(os.getcwd())
step = 0
before = 0

all_out = []
args = []
scatteroptions= []

start = 525
end   = 550

count=1
for files in os.listdir():
    #args = []
    #scatteroptions= []
    
    keyword = 'Ti 0nm (24nm linescan6)'
    if (( keyword in files) and ('.s' in files)  ): #and ('.s' in files)
        
        original_name = keyword
        normalized_name = original_name+'_N'
        SG_name = original_name+'_SG'
        SS_name = original_name#+'_SS'
        
        new.read_input(original_name, files)
        
        dispersion = new.info[original_name][1,0] - new.info[original_name][0,0]
        loc_start = int(np.min(np.where(np.absolute(new.info[original_name][:,0]-start)<dispersion)))
        loc_end = int(np.min(np.where(np.absolute(new.info[original_name][:,0]-(end+1))<dispersion)))
        
        temp = new.info[original_name][loc_start:loc_end,:]
        new.info[original_name] = copy.deepcopy(temp)
        
        new.normalize(original_name,normalized_name,4)
        new.normalize(normalized_name,normalized_name,1)
        
        #step += np.max(0.5*new.info[normalized_name][:,1])
        #new.info[normalized_name][:,1] += step
        
        new.savitzky_golay(normalized_name, SG_name,'start','end',5,3,0)
        new.smoothing_spline(SG_name,SS_name,start,end,0.99998)  # Ru:0.99999
        
        step += 0.5*(np.max(new.info[normalized_name][:,1])-np.min(new.info[normalized_name][:,1]))
        
        #offset = np.min(new.info[normalized_name][:,1])
        #new.info[normalized_name][:,1] -= offset
        #new.info[SS_name][:,1] -= offset
        #new.info[SG_name][:,1] -= offset
        
        new.info[normalized_name][:,1] += before #step
        new.info[SS_name][:,1] += before #step
        new.info[SG_name][:,1] += before #step
        
        before = 0.4*count#*np.max(new.info[normalized_name][:,1])
        count += 1
        
        
        args.append(normalized_name)
        #args.append(SG_name)
        args.append(SS_name)
        
        scatteroptions.append(False)
        #scatteroptions.append(False)
        scatteroptions.append(False)
        
        #new.output(normalized_name,normalized_name)
        #new.output(SG_name,SG_name)
        #new.output(SS_name,SS_name)
        
        all_out.append(normalized_name)
        all_out.append(SS_name)
        
        #new.plot(args,scatteroptions,files+'.png',xlim=[start,end],ylim=[0.0002,0.0008])
        
        
new.plot(args,scatteroptions,keyword+'.png',xlim=[start,end], legendout=False)
all_out.append(keyword) 
new.all_output(all_out)