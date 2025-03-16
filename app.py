from flask import Flask, render_template, url_for, redirect, request
import matplotlib
import io
import vercel_blob
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)


#CPU Scheduling Algorithms
def fcfs_scheduling(arrival_time, burst_time):
    n = len(arrival_time)
    completion_time = [0] * n
    turn_around_time = [0] * n
    waiting_time = [0] * n

    processes = sorted(enumerate(arrival_time), key=lambda x: x[1])
    sorted_indices = [p[0] for p in processes]

    current_time = 0

    for i in sorted_indices:
        if current_time < arrival_time[i]:
            current_time = arrival_time[i]

        completion_time[i] = current_time + burst_time[i]
        turn_around_time[i] = completion_time[i] - arrival_time[i]
        waiting_time[i] = turn_around_time[i] - burst_time[i]

        current_time = completion_time[i]

    total_tat = sum(turn_around_time)
    total_wt = sum(waiting_time)
    avg_tat = round(total_tat / n,2)
    avg_wt = round(total_wt / n,2)

    result= {
        'algorithm':'First Come First Served',
        'arrival time': arrival_time,
        'burst time': burst_time,
        'ct': completion_time,
        'tat': turn_around_time,
        'wt': waiting_time,
        'total tat': total_tat,
        'total wt': total_wt,
        'avg tat': avg_tat,
        'avg wt': avg_wt
    }

    return result

def sjf_scheduling(arrival_time, burst_time):
    n = len(arrival_time)

    completion_time = [0] * n
    turn_around_time = [0] * n
    waiting_time = [0] * n
    remaining_processes = list(range(n))

    current_time = 0

    while remaining_processes:
        available = [i for i in remaining_processes if arrival_time[i] <= current_time]
        
        if not available:  
            current_time = min(arrival_time[i] for i in remaining_processes)
            continue

        next_process = min(available, key=lambda i: burst_time[i])
        
        completion_time[next_process] = current_time + burst_time[next_process]
        turn_around_time[next_process] = completion_time[next_process] - arrival_time[next_process]
        waiting_time[next_process] = turn_around_time[next_process] - burst_time[next_process]
        
        current_time = completion_time[next_process]
        remaining_processes.remove(next_process)

    total_tat = sum(turn_around_time)
    total_wt = sum(waiting_time)
    avg_tat = round(total_tat / n, 2)
    avg_wt = round(total_wt / n, 2)

    return {
        'algorithm': 'Shortest Job First',
        'arrival time': arrival_time,
        'burst time': burst_time,
        'ct': completion_time,
        'tat': turn_around_time,
        'wt': waiting_time,
        'total tat': total_tat,
        'total wt': total_wt,
        'avg tat': avg_tat,
        'avg wt': avg_wt
    }


def srtf_scheduling(arrival_time, burst_time):
    n = len(arrival_time)
    remaining_time = burst_time[:]
    completion_time = [0] * n
    turn_around_time = [0] * n
    waiting_time = [0] * n

    current_time = 0
    completed = 0
    while completed < n:
        idx = -1
        min_bt = float('inf')
        for i in range(n):
            if arrival_time[i] <= current_time and remaining_time[i] > 0:
                if remaining_time[i] < min_bt or (remaining_time[i] == min_bt and i < idx):
                    min_bt = remaining_time[i]
                    idx = i

        if idx == -1:
            current_time += 1
            continue


        remaining_time[idx] -= 1
        current_time += 1

        if remaining_time[idx] == 0:
            completed += 1
            completion_time[idx] = current_time
            turn_around_time[idx] = completion_time[idx] - arrival_time[idx]
            waiting_time[idx] = turn_around_time[idx] - burst_time[idx]

    total_tat = sum(turn_around_time)
    total_wt = sum(waiting_time)
    avg_tat = round(total_tat / n,2)
    avg_wt = round(total_wt / n,2)

    return {
        'algorithm':'Shortest Remaining Time First',
        'arrival time': arrival_time,
        'burst time': burst_time,
        'ct': completion_time,
        'tat': turn_around_time,
        'wt': waiting_time,
        'total tat': total_tat,
        'total wt': total_wt,
        'avg tat': avg_tat,
        'avg wt': avg_wt
    }


def rr_scheduling(arrival_time, burst_time, quantum):
    n = len(arrival_time)
    remaining_time = burst_time[:]
    completion_time = [0] * n
    turn_around_time = [0] * n
    waiting_time = [0] * n

    current_time = 0
    queue = []

    while any(rt > 0 for rt in remaining_time):
        for i in range(n):
            if arrival_time[i] <= current_time and remaining_time[i] > 0 and i not in queue:
                queue.append(i)

        if not queue:
            current_time += 1
            continue

        idx = queue.pop(0)
        execution_time = min(quantum, remaining_time[idx])
        current_time += execution_time
        remaining_time[idx] -= execution_time

        if remaining_time[idx] == 0:
            completion_time[idx] = current_time
            turn_around_time[idx] = completion_time[idx] - arrival_time[idx]
            waiting_time[idx] = turn_around_time[idx] - burst_time[idx]

        for i in range(n):
            if arrival_time[i] <= current_time and remaining_time[i] > 0 and i not in queue:
                queue.append(i)

        if remaining_time[idx] > 0:
            queue.append(idx)

    total_tat = sum(turn_around_time)
    total_wt = sum(waiting_time)
    avg_tat = round(total_tat / n, 2)
    avg_wt = round(total_wt / n, 2)

    result = {
        'algorithm': 'Round Robin',
        'arrival time': arrival_time,
        'burst time': burst_time,
        'ct': completion_time,
        'tat': turn_around_time,
        'wt': waiting_time,
        'total tat': total_tat,
        'total wt': total_wt,
        'avg tat': avg_tat,
        'avg wt': avg_wt
    }

    return result

def rr_scheduling(arrival_time, burst_time, quantum):
    n = len(arrival_time)
    remaining_time = burst_time[:]
    completion_time = [0] * n
    turn_around_time = [0] * n
    waiting_time = [0] * n
    time = 0
    queue = []
    visited = [False] * n

    while any(rt > 0 for rt in remaining_time):
        for i in range(n):
            if arrival_time[i] <= time and not visited[i] and remaining_time[i] > 0:
                queue.append(i)
                visited[i] = True

        if not queue: 
            time = min([arrival_time[i] for i in range(n) if remaining_time[i] > 0])
            continue

        idx = queue.pop(0)

        execution_time = min(quantum, remaining_time[idx])
        time += execution_time
        remaining_time[idx] -= execution_time

        for i in range(n):
            if arrival_time[i] > time - execution_time and arrival_time[i] <= time and not visited[i] and remaining_time[i] > 0:
                queue.append(i)
                visited[i] = True

        if remaining_time[idx] == 0:
            completion_time[idx] = time
            turn_around_time[idx] = completion_time[idx] - arrival_time[idx]
            waiting_time[idx] = turn_around_time[idx] - burst_time[idx]
        else:
            queue.append(idx)

    total_tat = sum(turn_around_time)
    total_wt = sum(waiting_time)
    avg_tat = round(total_tat / n, 2)
    avg_wt = round(total_wt / n, 2)

    result = {
        'algorithm': 'Round Robin',
        'arrival time': arrival_time,
        'burst time': burst_time,
        'ct': completion_time,
        'tat': turn_around_time,
        'wt': waiting_time,
        'total tat': total_tat,
        'total wt': total_wt,
        'avg tat': avg_tat,
        'avg wt': avg_wt
    }

    return result

def priority_non_preemptive_scheduling(arrival_time, burst_time, priority):
    n = len(arrival_time)

    completion_time = [0] * n
    turn_around_time = [0] * n
    waiting_time = [0] * n
    gantt_chart = []

    processes = list(enumerate(zip(arrival_time, burst_time, priority)))
    
    current_time = 0
    completed = 0

    while completed < n:
        available_processes = [i for i in range(n) if arrival_time[i] <= current_time and completion_time[i] == 0]
        
        if available_processes:
            idx = min(available_processes, key=lambda x: priority[x])

            completion_time[idx] = current_time + burst_time[idx]
            turn_around_time[idx] = completion_time[idx] - arrival_time[idx]
            waiting_time[idx] = turn_around_time[idx] - burst_time[idx]

            gantt_chart.append((f"P{idx + 1}", current_time, completion_time[idx]))

            current_time = completion_time[idx]
            completed += 1
        else:
            next_arrival = min([arrival_time[i] for i in range(n) if completion_time[i] == 0], default=current_time)
            current_time = next_arrival

    total_tat = sum(turn_around_time)
    total_wt = sum(waiting_time)
    avg_tat = round(total_tat / n, 2)
    avg_wt = round(total_wt / n, 2)


    result = {
        'algorithm': 'Priority Non-Preemptive',
        'arrival time': arrival_time,
        'burst time': burst_time,
        'priority': priority,
        'ct': completion_time,
        'tat': turn_around_time,
        'wt': waiting_time,
        'total tat': total_tat,
        'total wt': total_wt,
        'avg tat': avg_tat,
        'avg wt': avg_wt
    }

    return result



def priority_preemptive_scheduling(arrival_time, burst_time, priority):
    n = len(arrival_time)

    remaining_burst_time = burst_time.copy()
    completion_time = [0] * n
    turn_around_time = [0] * n
    waiting_time = [0] * n

    processes = list(enumerate(zip(arrival_time, remaining_burst_time, priority)))
    processes.sort(key=lambda x: (x[1][0], -x[1][2]))

    current_time = 0
    completed = 0
    while completed < n:
        available_processes = [p for p in processes if p[1][0] <= current_time and remaining_burst_time[p[0]] > 0]
        
        if available_processes:
            current_process = min(available_processes, key=lambda x: (-x[1][2], x[1][0]))
            idx = current_process[0]
            
            current_time += 1
            remaining_burst_time[idx] -= 1

            if remaining_burst_time[idx] == 0:
                completion_time[idx] = current_time
                turn_around_time[idx] = completion_time[idx] - arrival_time[idx]
                waiting_time[idx] = turn_around_time[idx] - burst_time[idx]
                completed += 1
        else:
            current_time += 1

    total_tat = sum(turn_around_time)
    total_wt = sum(waiting_time)
    avg_tat = round(total_tat / n,2)
    avg_wt = round(total_wt / n,2)

    result = {
        'algorithm':'Priority Preemptive',
        'arrival time': arrival_time,
        'burst time': burst_time,
        'priority': priority,
        'ct': completion_time,
        'tat': turn_around_time,
        'wt': waiting_time,
        'total tat': total_tat,
        'total wt': total_wt,
        'avg tat': avg_tat,
        'avg wt': avg_wt
    }

    return result

#Memory Management algorithms
def first_fit(partitions, processes):
    step_part=[]
    step_part.append(partitions.copy())
    allocation = [-1] * len(processes) 
    for i, process in enumerate(processes):
        for j in range(len(partitions)):
            if process <= partitions[j]:
                allocation[i] = j
                partitions[j] -= process 
                step_part.append(partitions.copy())
                break

    total_free = sum(partitions)
    not_allocated = [processes[i] for i, a in enumerate(allocation) if a == -1]

    return {
        "algorithm":"First Fit",
        "step_partition":step_part,
        "total_free": total_free,
        "not_allocated": not_allocated,
        "processes":processes,
        "length":len(processes),
        "memory":partitions
    }

def best_fit(partitions, processes):
    step_part=[]
    step_part.append(partitions.copy())
    allocation = [-1] * len(processes)
    for i, process in enumerate(processes):
        best_idx = -1
        for j in range(len(partitions)):
            if process <= partitions[j]:
                if best_idx == -1 or partitions[j] < partitions[best_idx]:
                    best_idx = j
        if best_idx != -1:
            allocation[i] = best_idx
            partitions[best_idx] -= process
            step_part.append(partitions.copy())

    total_free = sum(partitions)
    not_allocated = [processes[i] for i in range(len(processes)) if allocation[i] == -1]

    return {
        "algorithm":"Best Fit",
        "step_partition":step_part,
        "total_free": total_free,
        "not_allocated": not_allocated,
        "processes":processes,
        "length":len(processes),
        "memory":partitions
    }


def worst_fit(partitions, processes):
    step_part=[]
    step_part.append(partitions.copy())
    allocation = [-1] * len(processes)
    for i, process in enumerate(processes):
        worst_idx = -1
        for j in range(len(partitions)):
            if process <= partitions[j]:
                if worst_idx == -1 or partitions[j] > partitions[worst_idx]:
                    worst_idx = j
        if worst_idx != -1:
            allocation[i] = worst_idx
            partitions[worst_idx] -= process
            step_part.append(partitions.copy())

    total_free = sum(partitions)
    not_allocated = [processes[i] for i in range(len(processes)) if allocation[i] == -1]

    return {
        "algorithm":"Worst Fit",
        "step_partition":step_part,
        "total_free": total_free,
        "not_allocated": not_allocated,
        "processes":processes,
        "length":len(processes),
        "memory":partitions
    }

#Page Replacement
def fifo_page_replacement(reference_string, num_frames):
    frames = []
    page_faults = 0
    page_hits = 0
    history = []
    pointer = 0 
    x=""
    for page in reference_string:
        if page in frames:
            page_hits += 1
            x="Hit"
        else:
            page_faults += 1
            x="Miss"
            if len(frames) < num_frames:
                frames.append(page)
            else:
                frames[pointer] = page
                pointer = (pointer + 1) % num_frames
        history.append([x,frames[:]])

    total_references = len(reference_string)
    hit_ratio = round(page_hits / total_references,2)
    miss_ratio = round(page_faults / total_references,2)

    result = {
        "reference":reference_string,
        "length":len(reference_string),
        "algorithm":"First in First Out",
        "hits": page_hits,
        "misses": page_faults,
        "hit_ratio": hit_ratio,
        "miss_ratio": miss_ratio,
        "frame_history": history
    }

    return result


def optimal_page_replacement(reference_string, num_frames):
    frames = []
    page_faults = 0
    page_hits = 0
    history = []
    x=""

    for i, page in enumerate(reference_string):
        if page in frames:
            page_hits += 1
            x="Hit"
        else:
            page_faults += 1
            x="Miss"
            if len(frames) < num_frames:
                frames.append(page)
            else:
                future_indices = [reference_string[i+1:].index(f) if f in reference_string[i+1:] else float('inf') for f in frames]
                victim_index = future_indices.index(max(future_indices))
                frames[victim_index] = page

        history.append([x,frames[:]])

    total_references = len(reference_string)
    hit_ratio = round(page_hits / total_references,2)
    miss_ratio = round(page_faults / total_references,2)

    result = {
        "reference":reference_string,
        "length":len(reference_string),
        "algorithm":"Optimal",
        "hits": page_hits,
        "misses": page_faults,
        "hit_ratio": hit_ratio,
        "miss_ratio": miss_ratio,
        "frame_history": history
    }

    return result

def lru_page_replacement(reference_string, num_frames):
    frames = []
    page_faults = 0
    page_hits = 0
    history = []
    recent_usage = {} 
    x=""

    for current_time, page in enumerate(reference_string):
        if page in frames:
            page_hits += 1
            x="Hit"
            recent_usage[page] = current_time
        else:
            page_faults += 1
            x="Miss"
            if len(frames) < num_frames:
                frames.append(page)
            else:
                lru_page = min(frames, key=lambda x: recent_usage[x])
                lru_index = frames.index(lru_page)
                frames[lru_index] = page
            recent_usage[page] = current_time
        history.append([x,frames[:]])

    total_references = len(reference_string)
    hit_ratio = round(page_hits / total_references,2)
    miss_ratio = round(page_faults / total_references,2)

    result = {
        "reference":reference_string,
        "length":len(reference_string),
        "algorithm":"Least Recently Used",
        "hits": page_hits,
        "misses": page_faults,
        "hit_ratio": hit_ratio,
        "miss_ratio": miss_ratio,
        "frame_history": history
    }

    return result

#disk scheduling 
import matplotlib.pyplot as plt

def fcfs_disk_scheduling(requests, start, disk_size):
    head = start
    seek_sequence = [head] + requests
    total_seek = sum(abs(seek_sequence[i] - seek_sequence[i - 1]) for i in range(1, len(seek_sequence)))
    
    return {
        "seek_sequence": seek_sequence,
        "total_seek": total_seek,
        "algorithm": "First Come First Served"
    }

def sstf_disk_scheduling(requests, start, disk_size):
    head = start
    remaining_requests = requests[:]
    seek_sequence = []
    total_seek = 0

    while remaining_requests:
        closest_request = min(remaining_requests, key=lambda r: abs(r - head))
        total_seek += abs(closest_request - head)
        head = closest_request
        seek_sequence.append(head)
        remaining_requests.remove(closest_request)

    return {
        "seek_sequence": seek_sequence,
        "total_seek": total_seek,
        "algorithm": "Shortest Seek Time First"
    }

def scan_disk_scheduling(requests, start, disk_size):
    requests = sorted(requests)
    seek_sequence = []
    total_seek = 0

    left = [r for r in requests if r < start]
    right = [r for r in requests if r >= start]

    seek_sequence.extend(right)
    seek_sequence.append(disk_size - 1) 

    if left:
        seek_sequence.extend(reversed(left))
        seek_sequence.append(min(left))

    current_position = start
    for position in seek_sequence:
        total_seek += abs(position - current_position)
        current_position = position

    return {
        "seek_sequence":seek_sequence,
        "total_seek": total_seek,
        "algorithm": "Scan"
    }

def c_scan_disk_scheduling(requests, start, disk_size):
    requests = sorted(requests)
    seek_sequence = []
    total_seek = 0

    left = [r for r in requests if r < start]
    right = [r for r in requests if r >= start]

    seek_sequence.extend(right)
    seek_sequence.append(disk_size - 1) 

    seek_sequence.append(0)

    if left:
        seek_sequence.extend(left)

    current_position = start
    for position in seek_sequence:
        total_seek += abs(position - current_position)
        current_position = position

    return {
        "seek_sequence":  seek_sequence,
        "total_seek": total_seek,
        "algorithm": "C-Scan"
    }


def look_disk_scheduling(requests, start, disk_size):
    requests = sorted(requests)
    seek_sequence = []
    total_seek = 0

    left = [r for r in requests if r < start]
    right = [r for r in requests if r >= start]

    seek_sequence.extend(right)

    if left:
        seek_sequence.extend(reversed(left))

    current_position = start
    for position in seek_sequence:
        total_seek += abs(position - current_position)
        current_position = position

    return {
        "seek_sequence": seek_sequence,
        "total_seek": total_seek,
        "algorithm": "Look"
    }



def c_look_disk_scheduling(requests, start, disk_size):
    requests = sorted(requests)
    seek_sequence = []
    total_seek = 0

    left = [r for r in requests if r < start]
    right = [r for r in requests if r >= start]

    seek_sequence.extend(right)
    seek_sequence.extend(left)

    current_position = start
    for position in seek_sequence:
        total_seek += abs(position - current_position)
        current_position = position

    return {
        "seek_sequence":seek_sequence,
        "total_seek": total_seek,
        "algorithm": "C-Look"
    }

def createYticks(length):
    val=[]
    try:
        for i in range(0,length):
            st="Step-"+str(i)
            val.append(st)
        return val
    except:
        return list(range(1,length))
        
def visualize_disk_scheduling(algorithm_name, result, disk_size):
    seek_sequence = result["seek_sequence"]
    total_seek = result["total_seek"]
    disk_size-=1
    plt.figure(figsize=(10, 4))
    plt.plot(seek_sequence, [i for i in range(len(seek_sequence))], marker="o", linestyle="-", color="b")
    plt.xlim(0, disk_size)
    plt.grid(True)

    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')

    plt.gca().invert_yaxis()
    plt.xticks(seek_sequence)
    
    plt.axvline(x=0, color='r', linestyle='--', linewidth=5, label='Start (0)')
    plt.axvline(x=disk_size, color='g', linestyle='--', linewidth=5, label=f'Disk Size ({disk_size})')
    arr=createYticks(len(seek_sequence))
    plt.yticks(range(0,len(seek_sequence)),arr)
    plt.legend()
    plt.savefig("/static/data/disk-plot.png")



def convertToArray(text):
    arr = []
    text=text.split(" ")
    for i in text:
        if i.isdigit():
            arr.append(int(i))
    return arr


#format the input and pass it to the desired function
def getResultForCpuScheduling(algorithm,at,bt,pt='0',qt=' 0'):
    try:
        at=convertToArray(at)
        bt=convertToArray(bt)
        pt=convertToArray(pt)
        
        
        if algorithm not in ['option-5','option-6']:
            if len(at)!=len(bt):
                raise Exception
        else:
            if len(at)!=len(bt)!=len(pt):
                raise Exception
            
        if algorithm=="option-1":
            return fcfs_scheduling(at,bt)
        elif algorithm=="option-2":
            return sjf_scheduling(at,bt)
        elif algorithm=="option-3":
            return srtf_scheduling(at,bt)
        elif algorithm=="option-4":
            return rr_scheduling(at,bt,int(qt))
        elif algorithm=="option-5":
            return priority_preemptive_scheduling(at,bt,pt)
        else:
            return priority_non_preemptive_scheduling(at,bt,pt)
        
    except:
        return "Invalid input"
    

def getResultForMemoryManagement(algorithm,partition,process):
    try:
        partition=convertToArray(partition)
        process=convertToArray(process)
        
            
        if algorithm=="option-1":
            return first_fit(partition,process)
        elif algorithm=="option-2":
            return best_fit(partition,process)
        else:
            return worst_fit(partition,process)
        
    except:
        return "Invalid input"

def getResultForPageReplacement(algorithm,reference,frame):
    try:
        reference=convertToArray(reference)
        frame=int(frame)

        if algorithm=="option-1":
            return fifo_page_replacement(reference,frame)
        elif algorithm=="option-2":
            return optimal_page_replacement(reference,frame)
        else:
            return lru_page_replacement(reference,frame)
    except:
        return "Invalid input"

    
def getResultForDiskScheduling(algorithm,reference,head,disk_size):
    try:
        reference=convertToArray(reference)
        head=int(head)
        disk_size=int(disk_size)

        if algorithm=="option-1":
            result= fcfs_disk_scheduling(reference,head,disk_size)
            visualize_disk_scheduling("First Come First Served",result,disk_size)
            return result
        
        elif algorithm=="option-2":
            result= sstf_disk_scheduling(reference,head,disk_size)
            visualize_disk_scheduling("Shortest Seek Time First",result,disk_size)
            return result
        
        elif algorithm=="option-3":
            result= scan_disk_scheduling(reference,head,disk_size)
            visualize_disk_scheduling("Scan",result,disk_size)
            return result
        
        elif algorithm=="option-4":
            result= c_scan_disk_scheduling(reference,head,disk_size)
            visualize_disk_scheduling("C-Scan",result,disk_size)
            return result
        
        elif algorithm=="option-5":
            result= look_disk_scheduling(reference,head,disk_size)
            visualize_disk_scheduling("Lookup",result,disk_size)
            return result
        
        else:
            result= c_look_disk_scheduling(reference,head,disk_size)
            visualize_disk_scheduling("C-Lookup",result,disk_size)
            return result
        
    except:
        return "Invalid input"
    
        

@app.route("/")
def home_redirect():
    return redirect(url_for('index'))


@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/cpu_scheduling")
def cpu_scheduling():
    return render_template("cpu_scheduling.html")

@app.route("/memory_management")
def memory_management():
    return render_template("memory_management.html")

@app.route("/page_replacement")
def page_replacement():
    return render_template("page_replacement.html")

@app.route("/disk_scheduling")
def disk_scheduling():
    return render_template("disk_scheduling.html")


@app.route("/cpu_scheduling",methods=['POST','GET'])
def input_cpu_scheduling():
    if(request.method=="POST"):
        algorithm = request.form.get('algorithm')
        at = request.form.get('at')
        bt = request.form.get('bt')
        pt = request.form.get("priority")
        qt = request.form.get("qt")
        data=getResultForCpuScheduling(algorithm,at,bt,pt,qt)

        return render_template("cpu_scheduling.html",data=data,algorithm=algorithm,at=at,bt=bt,pt=pt,qt=qt)
    return render_template("cpu_scheduling.html")

@app.route("/memory_management",methods=['POST','GET'])
def input_memeory_management():
    if(request.method=="POST"):
        algorithm=request.form.get('algorithm')
        partition=request.form.get('Partitions')
        process=request.form.get('Processes')
        data = getResultForMemoryManagement(algorithm,partition,process)

        return render_template("memory_management.html",data=data,algorithm=algorithm,partition=partition,process=process)
    return render_template("memory_management.html")

@app.route("/page_replacement",methods=['POST','GET'])
def input_page_replacement():
    if(request.method=="POST"):
        algorithm=request.form.get('algorithm')
        reference=request.form.get('reference')
        frame=request.form.get('frame')
        data = getResultForPageReplacement(algorithm,reference,frame)

        return render_template("page_replacement.html",data=data,algorithm=algorithm,reference=reference,frame=frame)
    return render_template("page_replacement.html")


@app.route("/disk_scheduling",methods=['POST','GET'])
def input_disk_scheduling():
    if(request.method=="POST"):
        algorithm=request.form.get('algorithm')
        reference=request.form.get('reference')
        head=request.form.get('head')
        disk_size=request.form.get('disk_size')
        data=getResultForDiskScheduling(algorithm,reference,head,disk_size)

        return render_template("disk_scheduling.html",data=data,algorithm=algorithm,reference=reference,head=head,disk_size=disk_size)
    return render_template("disk_scheduling.html")


if __name__=="__main__":
    app.run()
