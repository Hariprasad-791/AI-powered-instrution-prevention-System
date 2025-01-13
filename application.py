from flask_socketio import SocketIO, emit
from flask import Flask, render_template, url_for, copy_current_request_context, request
from random import random
from time import sleep
from threading import Thread, Event

from scapy.sendrecv import sniff

from flow.Flow import Flow
from flow.PacketInfo import PacketInfo

import numpy as np
import pickle
import csv 
import traceback

import json
import pandas as pd

# from models.AE import *

from scipy.stats import norm

import ipaddress
from urllib.request import urlopen

from tensorflow import keras

from lime import lime_tabular

import dill

import joblib

import plotly
import plotly.graph_objs

import warnings
warnings.filterwarnings("ignore")

def ipInfo(addr=''):
    try:
        if addr == '':
            url = 'https://ipinfo.io/json'
        else:
            url = 'https://ipinfo.io/' + addr + '/json'
        res = urlopen(url)
        #response from url(if res==None then check connection)
        data = json.load(res)
        #will load the json response into data
        return data['country']
    except Exception:
        return None
__author__ = 'hoang'


app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True

#turn the flask app into a socketio app
socketio = SocketIO(app, async_mode=None, logger=True, engineio_logger=True)

#random result Generator Thread
thread = Thread()
thread_stop_event = Event()

f = open("output_logs.csv", 'w')
w = csv.writer(f)
f2 = open("input_logs.csv", 'w')
w2 = csv.writer(f2)
 

cols = ['FlowID',
'FlowDuration',
'BwdPacketLenMax',
'BwdPacketLenMin',
'BwdPacketLenMean',
'BwdPacketLenStd',
'FlowIATMean',
'FlowIATStd',
'FlowIATMax',
'FlowIATMin',
'FwdIATTotal',
'FwdIATMean',
'FwdIATStd',
'FwdIATMax',
'FwdIATMin',
'BwdIATTotal',
'BwdIATMean',
'BwdIATStd',
'BwdIATMax',
'BwdIATMin',
'FwdPSHFlags',
'FwdPackets_s',
'MaxPacketLen',
'PacketLenMean',
'PacketLenStd',
'PacketLenVar',
'FINFlagCount',
'SYNFlagCount',
'PSHFlagCount',
'ACKFlagCount',
'URGFlagCount',
'AvgPacketSize',
'AvgBwdSegmentSize',
'InitWinBytesFwd',
'InitWinBytesBwd',
'ActiveMin',
'IdleMean',
'IdleStd',
'IdleMax',
'IdleMin',
'Src',
'SrcPort',
'Dest',
'DestPort',
'Protocol',
'FlowStartTime',
'FlowLastSeen',
'PName',
'PID',
'Classification',
'Probability',
'Risk']

ae_features = np.array(['FlowDuration',
'BwdPacketLengthMax',
'BwdPacketLengthMin',
'BwdPacketLengthMean',
'BwdPacketLengthStd',
'FlowIATMean',
'FlowIATStd',
'FlowIATMax',
'FlowIATMin',
'FwdIATTotal',
'FwdIATMean',
'FwdIATStd',
'FwdIATMax',
'FwdIATMin',
'BwdIATTotal',
'BwdIATMean',
'BwdIATStd',
'BwdIATMax',
'BwdIATMin',
'FwdPSHFlags',
'FwdPackets/s',
'PacketLengthMax',
'PacketLengthMean',
'PacketLengthStd',
'PacketLengthVariance',
'FINFlagCount',
'SYNFlagCount',
'PSHFlagCount',
'ACKFlagCount',
'URGFlagCount',
'AveragePacketSize',
'BwdSegmentSizeAvg',
'FWDInitWinBytes',
'BwdInitWinBytes',
'ActiveMin',
'IdleMean',
'IdleStd',
'IdleMax',
'IdleMin'])

flow_count = 0
flow_df = pd.DataFrame(columns =cols)

src_ip_dict = {}

current_flows = {}
FlowTimeout = 600

# Initialize the blocked IPs set
blocked_ips = set()
ae_scaler = joblib.load("models/preprocess_pipeline_AE_39ft.save")
ae_model = keras.models.load_model('models/autoencoder_39ft.hdf5')

with open('models/model.pkl', 'rb') as f:
    classifier = pickle.load(f)

with open('models/explainer', 'rb') as f:
    explainer = dill.load(f)
predict_fn_rf = lambda x: classifier.predict_proba(x).astype(float)

def classify(features):
    # preprocess
    global flow_count
    feature_string = [str(i) for i in features[39:]]
    record = features.copy()
    features = [np.nan if x in [np.inf, -np.inf] else float(x) for x in features[:39]]
    

    if feature_string[0] in src_ip_dict.keys():
        src_ip_dict[feature_string[0]] +=1
    else:
        src_ip_dict[feature_string[0]] = 1

    for i in [0,2]:
        ip = feature_string[i] #feature_string[0] is src, [2] is dst
        if not ipaddress.ip_address(ip).is_private:
            country = ipInfo(ip)
            if country is not None and country not in  ['ano', 'unknown']:
                img = ' <img src="static/images/blank.gif" class="flag flag-' + country.lower() + '" title="' + country + '">'
            else:
                img = ' <img src="static/images/blank.gif" class="flag flag-unknown" title="UNKNOWN">'
        else:
            img = ' <img src="static/images/lan.gif" height="11px" style="margin-bottom: 0px" title="LAN">'
        feature_string[i]+=img

    if np.nan in features:
        return

    # features = normalisation.transform([features])
    result = classifier.predict([features])
    proba = predict_fn_rf([features])
    proba_score = [proba[0].max()]
    proba_risk = sum(list(proba[0,1:]))
    if proba_risk >0.8: risk = ["<p style=\"color:red;\">Very High</p>"]
    elif proba_risk >0.6: risk = ["<p style=\"color:orangered;\">High</p>"]
    if proba_risk >0.4: risk = ["<p style=\"color:orange;\">Medium</p>"]
    if proba_risk >0.2: risk = ["<p style=\"color:green;\">Low</p>"]
    else: risk = ["<p style=\"color:limegreen;\">Minimal</p>"]

    # x = K.process(features[0])
    # z_scores = round((x-m)/s,2)
    # p_values = norm.sf(abs(z_scores))*2


    classification = [str(result[0])]
    if result != 'Benign':
        print(feature_string + classification + proba_score )

    flow_count +=1
    w.writerow(['Flow #'+str(flow_count)] )
    w.writerow(['Flow info:']+feature_string)
    w.writerow(['Flow features:']+features)
    w.writerow(['Prediction:']+classification+ proba_score)
    w.writerow(['--------------------------------------------------------------------------------------------------'])

    w2.writerow(['Flow #'+str(flow_count)] )
    w2.writerow(['Flow info:']+features)
    w2.writerow(['--------------------------------------------------------------------------------------------------'])
    flow_df.loc[len(flow_df)] = [flow_count]+ record + classification + proba_score + risk


    ip_data = {'SourceIP': src_ip_dict.keys(), 'count': src_ip_dict.values()} 
    ip_data= pd.DataFrame(ip_data)
    ip_data=ip_data.to_json(orient='records')


    socketio.emit('newresult', {'result':[flow_count]+ feature_string + classification + proba_score + risk, "ips": json.loads(ip_data)}, namespace='/test')
    # socketio.emit('newresult', {'result': feature_string + classification}, namespace='/test')
    return [flow_count]+ record + classification+ proba_score + risk

def newPacket(p):
    try:
        packet = PacketInfo()
        packet.setSrc(p)
        src_ip = packet.getSrc()

        # Drop packets from blocked IPs
        if src_ip in blocked_ips:
            print(f"Packet from {src_ip} dropped (blocked).")
            return  # Skip further processing of this packet

        # Continue processing the packet
        packet.setDest(p)
        packet.setSrcPort(p)
        packet.setDestPort(p)
        packet.setProtocol(p)
        packet.setTimestamp(p)
        packet.setPSHFlag(p)
        packet.setFINFlag(p)
        packet.setSYNFlag(p)
        packet.setACKFlag(p)
        packet.setURGFlag(p)
        packet.setRSTFlag(p)
        packet.setPayloadBytes(p)
        packet.setHeaderBytes(p)
        packet.setPacketSize(p)
        packet.setWinBytes(p)
        packet.setFwdID()
        packet.setBwdID()

        # Handle flow logic (existing implementation)
        if packet.getFwdID() in current_flows.keys():
            flow = current_flows[packet.getFwdID()]
            # Handle timeout or FIN/RST flags
            if (packet.getTimestamp() - flow.getFlowLastSeen()) > FlowTimeout:
                classify(flow.terminated())
                del current_flows[packet.getFwdID()]
                flow = Flow(packet)
                current_flows[packet.getFwdID()] = flow
            elif packet.getFINFlag() or packet.getRSTFlag():
                flow.new(packet, 'fwd')
                classify(flow.terminated())
                del current_flows[packet.getFwdID()]
            else:
                flow.new(packet, 'fwd')
                current_flows[packet.getFwdID()] = flow
        elif packet.getBwdID() in current_flows.keys():
            flow = current_flows[packet.getBwdID()]
            if (packet.getTimestamp() - flow.getFlowLastSeen()) > FlowTimeout:
                classify(flow.terminated())
                del current_flows[packet.getBwdID()]
                flow = Flow(packet)
                current_flows[packet.getFwdID()] = flow
            elif packet.getFINFlag() or packet.getRSTFlag():
                flow.new(packet, 'bwd')
                classify(flow.terminated())
                del current_flows[packet.getBwdID()]
            else:
                flow.new(packet, 'bwd')
                current_flows[packet.getBwdID()] = flow
        else:
            flow = Flow(packet)
            current_flows[packet.getFwdID()] = flow

    except AttributeError:
        # Handle invalid packets
        return
    except Exception:
        traceback.print_exc()


def snif_and_detect():
    while not thread_stop_event.isSet():
        print("Begin Sniffing".center(20, ' '))
        sniff(prn=newPacket, store=False)  # Use store=False for performance
        for f in current_flows.values():
            classify(f.terminated())


@app.route('/toggle-block', methods=['POST'])
def toggle_block():
    data = request.get_json()
    src_ip = data.get('src_ip')
    if not src_ip:
        return {'message': 'Invalid IP'}, 400

    if src_ip in blocked_ips:
        blocked_ips.remove(src_ip)
        action = "unblocked"
    else:
        blocked_ips.add(src_ip)
        action = "blocked"

    return {'message': f'{src_ip} has been {action}.', 'status': action}, 200



@app.route('/')
def index():
    #only by sending this page first will the client be connected to the socketio instance
    return render_template('index.html')

def newPacket(p):
    try:
        packet = PacketInfo()
        packet.setSrc(p)
        src_ip = packet.getSrc()

        # Drop packets from blocked IPs
        if src_ip in blocked_ips:
            print(f"Packet from {src_ip} dropped (blocked).")
            return  # Skip further processing of this packet

        # Continue processing the packet
        packet.setDest(p)
        packet.setSrcPort(p)
        packet.setDestPort(p)
        packet.setProtocol(p)
        packet.setTimestamp(p)
        packet.setPSHFlag(p)
        packet.setFINFlag(p)
        packet.setSYNFlag(p)
        packet.setACKFlag(p)
        packet.setURGFlag(p)
        packet.setRSTFlag(p)
        packet.setPayloadBytes(p)
        packet.setHeaderBytes(p)
        packet.setPacketSize(p)
        packet.setWinBytes(p)
        packet.setFwdID()
        packet.setBwdID()

        # Handle flow logic (existing implementation)
        if packet.getFwdID() in current_flows.keys():
            flow = current_flows[packet.getFwdID()]
            # Handle timeout or FIN/RST flags
            if (packet.getTimestamp() - flow.getFlowLastSeen()) > FlowTimeout:
                classify(flow.terminated())
                del current_flows[packet.getFwdID()]
                flow = Flow(packet)
                current_flows[packet.getFwdID()] = flow
            elif packet.getFINFlag() or packet.getRSTFlag():
                flow.new(packet, 'fwd')
                classify(flow.terminated())
                del current_flows[packet.getFwdID()]
            else:
                flow.new(packet, 'fwd')
                current_flows[packet.getFwdID()] = flow
        elif packet.getBwdID() in current_flows.keys():
            flow = current_flows[packet.getBwdID()]
            if (packet.getTimestamp() - flow.getFlowLastSeen()) > FlowTimeout:
                classify(flow.terminated())
                del current_flows[packet.getBwdID()]
                flow = Flow(packet)
                current_flows[packet.getFwdID()] = flow
            elif packet.getFINFlag() or packet.getRSTFlag():
                flow.new(packet, 'bwd')
                classify(flow.terminated())
                del current_flows[packet.getBwdID()]
            else:
                flow.new(packet, 'bwd')
                current_flows[packet.getBwdID()] = flow
        else:
            flow = Flow(packet)
            current_flows[packet.getFwdID()] = flow

    except AttributeError:
        # Handle invalid packets
        return
    except Exception:
        traceback.print_exc()

@socketio.on('connect', namespace='/test')
def test_connect():
    # need visibility of the global thread object
    global thread
    print('Client connected')

    #Start the random result generator thread only if the thread has not been started before.
    if not thread.is_alive():
        print("Starting Thread")
        thread = socketio.start_background_task(snif_and_detect)

@socketio.on('disconnect', namespace='/test')
def test_disconnect():
    print('Client disconnected')


if __name__ == "__main__":
    app.run(debug=False)

