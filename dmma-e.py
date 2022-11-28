import pandas as pd
import pm4py
from csv import DictReader
import numpy as np
import threading
from queue import Queue
from sklearn import tree
from sklearn.tree import export_text
import regex as re
from collections import OrderedDict
from skmultiflow.drift_detection.adwin import ADWIN
import matplotlib.pyplot as plt


def stream_csv(file, queue):
    with open(file) as read_obj:
        csv_dict_reader = DictReader(read_obj)
        for row in csv_dict_reader:
            queue.put(row)

def find_dps(pn, dps_data):
    global hoeffding_trees
    global all_decision_trees
    global class_labels_encoded
    out_in = dict()
    dps_data = dict()
    decision = dict()
    for p in pn.arcs:
        st_arc = str(p)
        outgoing = re.search("(^[^->]*)", st_arc).group(1)
        ingoing = re.search("->(.*)", st_arc).group(1)
        if outgoing in out_in:
            out_in[outgoing].append(ingoing)
        else:
            out_in[outgoing] = [ingoing]
    for key in out_in:
        if len(out_in[key]) > 1:
            decision[key] = out_in[key]
    decision_points = dict()
    all_decision_trees = []
    for key in decision:
        for key1 in out_in:
            if key == out_in[key1][0]:
                decision_points[key1] = decision[key]

    for key, value in decision_points.items():
        dps_data[key] = OrderedDict()
    for key in decision_points:
        all_decision_trees.append([key, tree.DecisionTreeClassifier(random_state=0, max_depth=4)])
    return decision_points, dps_data

def make_dfg_from_stream(new_element, id, event):
    global events_dfg
    global old_event
    global case_activity
    global cleanup_window
    global lossy_reduce
    global counter_total

    counter_total += 1

    case_id = new_element[id]
    new_event = new_element[event]
    if case_id in case_activity:
        last_event = case_activity[case_id]
        case_activity[case_id] = new_event
        list_events = (last_event, new_event)
        if list_events in events_dfg:
            events_dfg[list_events] += 1
        else:
            events_dfg[list_events] = 1
    else:
        case_activity[case_id] = new_event
    changed = False
    if (counter_total % cleanup_window == 0):
        new_dict = dict()
        for k in events_dfg:
            events_dfg[k] -= 1
            if events_dfg[k]>0:
                new_dict[k] = events_dfg[k]
                changed = True
    if changed:
        events_dfg = new_dict
    if len(case_activity) > 10000:
        case_activity.popitem(last=False)

def mine_decision_rule(dps_data, id, timestamp, event):
    global all_decision_trees
    global grace_period
    dfs = []
    counter = 0
    for k, v in dps_data.items():
        X = pd.DataFrame.from_dict(v)
        X = X.transpose()
        dfs.append(X)
    for df in dfs:
            y_var = df["class"]
            for colname, coltype in df.dtypes.to_dict().items():
                if coltype == 'object': df[colname] = df[colname].astype(int, errors="ignore")
            for colname, coltype in df.dtypes.to_dict().items():
                if coltype == 'object': df[colname] = df[colname].astype(bool, errors="ignore")
            names = df.select_dtypes(include=np.number).columns.tolist()
            names = [n for n in names if n != "class" and n != id and n != event]
            if (len(names)<=0):
                return False
            X_var = df.loc[:, names]
            clf = tree.DecisionTreeClassifier(random_state=0, max_depth=4)
            clf = clf.fit(X_var, y_var)
            all_decision_trees[counter][1] = clf
            accuracy = clf.score(X_var, y_var)
            all_decision_trees[counter].append(accuracy)
            all_decision_trees[counter].append(1)
            all_decision_trees[counter].append(accuracy/1) #total accuracy, pos 3
            all_decision_trees[counter].append([accuracy])
            all_decision_trees[counter].append([accuracy])
            all_decision_trees[counter].append([])
            all_decision_trees[counter].append(0)
            adwin = ADWIN()
            all_decision_trees[counter].append(adwin) #counter, pos 9, adwin
            all_decision_trees[counter].append(grace_period)  # counter, pos 10, adwin window size

            r = export_text(clf, feature_names=names)
            print("Initial Mining - Decision Point Number ", counter, ": \n", r, "Accuracy:", accuracy, "\n")
            counter += 1
    return dfs

def remine_decision_rule(dps_data, counter,grace_period, id, timestamp, event):
    global all_decision_trees
    keys = list(dps_data.keys())
    key = keys[counter]
    values = dps_data[key]
    X = pd.DataFrame.from_dict(values)
    df = X.transpose()

    try:
        y_var = df["class"]
        for colname, coltype in df.dtypes.to_dict().items():
            if coltype == 'object': df[colname] = df[colname].astype(int, errors="ignore")
        for colname, coltype in df.dtypes.to_dict().items():
            if coltype == 'object': df[colname] = df[colname].astype(bool, errors="ignore")
        names = df.select_dtypes(include=np.number).columns.tolist()
        names = [n for n in names if n != "class" and n != id and n != event]
        X_var = df.loc[:, names]
        clf = tree.DecisionTreeClassifier(random_state=0, max_depth=4)
        clf = clf.fit(X_var, y_var)
        all_decision_trees[counter] = [key, clf]
        accuracy = clf.score(X_var, y_var)
        all_decision_trees[counter].append(accuracy)
        all_decision_trees[counter].append(1)
        all_decision_trees[counter].append(accuracy/1)
        all_decision_trees[counter].append([accuracy])
        all_decision_trees[counter].append([accuracy])
        all_decision_trees[counter].append([])
        all_decision_trees[counter].append(0)

        adwin = ADWIN()
        all_decision_trees[counter].append(adwin)  # counter, pos 9, adwin
        all_decision_trees[counter].append(grace_period)  # counter, pos 10, adwin window size

        r = export_text(clf, feature_names=names)
        print(" Remining - Decision Point Number ", counter, ": \n", r, "Accuracy:", accuracy, "\n")
        counter += 1
    except Exception as e:
        print("Exception Mine Decision Rule: ", e)
        counter += 1
        pass

def check_dt_accuracy(new_element, current_event, counter, dps_data, iid, timestamp, event):
    global all_decision_trees
    global trace_dict
    global counter_total
    global adwin
    global accuracy_plot
    id = new_element[iid]
    dt = all_decision_trees[counter][1]
    grace_period = all_decision_trees[counter][10]
    y = np.array([current_event])
    col = []
    values = []
    all_data = trace_dict[id]
    for k1, v1 in all_data.items():
        if k1 == id:
            uuid = v1
        if k1 not in ["first_timestamp", "events", timestamp, iid]:
            if v1.isnumeric():
                col.append(k1)
                values.append(v1)
    X = pd.DataFrame(values, index=col).T
    try:
        accuracy = dt.score(X, y)
    except:
        remine_decision_rule(dps_data, counter,grace_period, iid, timestamp, event)
        return
    total_accuracy = all_decision_trees[counter][2]
    all_decision_trees[counter][2] += accuracy

    number_total = all_decision_trees[counter][3] + 1
    all_decision_trees[counter][3] = number_total
    mean_accuracy = round((accuracy + total_accuracy)/ number_total, 2)
    accuracy_plot.append(mean_accuracy)
    all_decision_trees[counter][4] = round(mean_accuracy, 2)
    all_decision_trees[counter][5].append(accuracy)
    all_decision_trees[counter][6].append(mean_accuracy)
    all_accuracy_scores = all_decision_trees[counter][5]
    if len(all_accuracy_scores) >= 20:
        all_accuracy_scores.pop(0)
    all_accuracy_scores = np.array(all_accuracy_scores)
    moving_average = float(sum(all_accuracy_scores)) / max(len(all_accuracy_scores), 1)
    all_decision_trees[counter][7].append(moving_average)
    adwin = all_decision_trees[counter][9]
    adwin.add_element(mean_accuracy)
    if adwin.detected_change():
        print("change detected", id, adwin.width)
        all_decision_trees[counter][9] = adwin
        all_decision_trees[counter][10] = adwin.width
        remine_decision_rule(dps_data, counter,grace_period, iid, timestamp, event)
    all_decision_trees[counter][8] += 1



def add_data_decision_elements(id, prev_element, current_event, dps_data, counter, timestamp):
    global all_decision_trees
    global no_rules
    for key, value in dps_data.items():
        if key == prev_element:
            dps_data[key][id] = dict()
            all_data = trace_dict[id]
            for k1, v1 in all_data.items():
                if k1 not in ["first_timestamp", "events", timestamp, id]:
                        dps_data[key][id][k1] = dict()
                        dps_data[key][id][k1] = v1
            dps_data[key][id]["class"] = current_event
            if (not no_rules):
                for k, v in dps_data.items():
                    leng = len(dps_data[k])
                    max_len = all_decision_trees[counter][10]
                    if leng >= max_len:
                        dps_data[k].popitem(last=False)
    return dps_data

def get_data_elements(new_element, decision_points, dps_data, id, timestamp, event):
    global trace_dict
    global no_rules
    iid = new_element[id]
    current_event = new_element[event]
    found = False
    if iid in trace_dict.keys():
        found = True
        prev_event = trace_dict[iid]["events"][-1]

        if decision_points != None and found:
            for key in decision_points:
                counter = list(decision_points).index(key)
                first = key
                second = decision_points[key]
                if first == prev_event and current_event in second:
                    if not no_rules:
                        check_dt_accuracy(new_element, current_event, counter, dps_data, id, timestamp, event)
                    dps_data = add_data_decision_elements(iid, prev_event, current_event, dps_data, counter, timestamp)
        trace_dict[iid]["events"].append(current_event)
        for key in new_element:
            if new_element[key] != "":
                trace_dict[iid][key] = new_element[key]
    else:
        trace_dict[iid] = {}
        trace_dict[iid]["first_timestamp"] = new_element[timestamp]
        trace_dict[iid]["events"] = []
        trace_dict[iid]["events"].append(current_event)
        if len(trace_dict)>1000:
            trace_dict.popitem(last=False)

    return dps_data

def make_heuristics_net(dfg):
    global old_pn
    new = False
    parameters = dict()
    parameters["dependency_treshold"] = 0.99
    parameters["and_treshold"] = 0.99
    parameters["loop_two_treshold"] = 0.99
    pn, im, fm = pm4py.algo.discovery.heuristics.variants.classic.apply_dfg(dfg, parameters=parameters)
    if str(old_pn) != str(pn):
        pm4py.view_petri_net(pn, im, fm)
        old_pn = pn
        new = True
    return pn, im, fm, new

def read_from_stream(n, queue):
    global counter_total
    global hoeffding_trees
    global counter_new
    global data_names
    decision_points = None
    global accuracy_plot
    global no_rules
    count = 0
    global grace_period
    grace_period = 200
    dps_data = dict()
    new = False
    file, id, event, timestamp = data_names
    while n:
        new_element = queue.get()
        make_dfg_from_stream(new_element, id, event)
        dps_data = get_data_elements(new_element, decision_points, dps_data, id, timestamp, event)
        if count % grace_period == 0:
            pn, im, fm, new = make_heuristics_net(events_dfg)
            if new:
                decision_points, dps_data = find_dps(pn, dps_data)
                print(decision_points)
                counter_new = 0
                no_rules = True
        if counter_new == grace_period and no_rules:
            print("Event number:", counter_total)
            print("Instance number:", new_element[id])
            dfs = mine_decision_rule(dps_data, id, timestamp, event)
            if dfs:
                no_rules = False
        counter_new += 1
        count += 1
        if queue.empty():
            n = False

def main():
    global data_names
    global accuracy_plot
    event_queue = Queue()
    file, id, event, timestamp = data_names
    worker1 = threading.Thread(target=stream_csv, args=(file, event_queue))
    worker1.start()
    worker2 = threading.Thread(target=read_from_stream, args=(1,event_queue))
    worker2.start()
    worker2.join()
    if not worker2.is_alive():
        plt.plot(accuracy_plot)
        plt.title('Average Accuracy - Synthetic Dataset')
        plt.ylabel('Accuracy')
        plt.axvline(2500, color='r') # vertical
        plt.legend(['Accuracy DP1', 'Accuracy DP2', "Accuracy DP3"], loc='lower left')
        plt.xlabel('Instance #')
        plt.show()

if __name__ == "__main__":
    class_labels_encoded= []
    events_dfg = dict()
    trace_dict = OrderedDict()
    case_activity = OrderedDict()
    hoeffding_trees = []
    all_decision_trees = []
    no_rules = True
    accuracy_plot = []
    cleanup_window = 50
    lossy_reduce = 2
    counter_total = 0
    counter_new = 0
    old_pn = ""
    old_event = "source"
    data_names = []
    use_case = "synthetic"

    if use_case == "synthetic":
        id = "uuid"
        event = "event"
        timestamp = "timestamp"
        print("FIRST DATASET - CONDITION CHANGES")
        file = "data/loan_dpa_runtime_changes_onlycondition.csv"
        data_names = [file, id, event, timestamp]
        main()

        print("SECOND DATASET - DATA CHANGES")
        file = "data/loan_dpa_runtime_changes_adddata.csv"
        main()

        print("THIRD DATASET - CLASSES CHANGE")
        file = "data/loan_dpa_runtime_changes_classes.csv"
        main()


        print("FOURTH DATASET - DPS CHANGE")
        file = "data/loan_dpa_runtime_changesdps.csv"
        main()


