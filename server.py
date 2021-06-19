# -*- coding: utf-8 -*-
import socket
import threading
import wave
import time
from BertSum.server_BertSum.bert_summary import Bertsum_pred
from tools.speech_t import speech_text
import os
import numpy as np
import json
import gc
# コマンドの定義
SET = 0
SUM = 1
WAV = 2
PLAY = 3
INPUT = 4
CON = 5
GIJI = 6

class StreamServer():
    def __init__(self, server_host, server_port):
        self.SERVER_HOST = server_host
        self.SERVER_PORT = int(server_port)
        self.CHUNK = 4048
        self.FORMAT = 8 # 16bit
        self.CHANNELS = 1             # monaural
        self.fs = 16000
        self.RATE = 2
        self.cla_dir = dict()

    def run(self):

        global addr

        # ソケットを生成しバインド
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.SERVER_HOST, self.SERVER_PORT))

        # コネクションの上限を5に設定し、リスニング開始
        server.listen(5)

        print('Server started on %s:%d' % (self.SERVER_HOST, self.SERVER_PORT))

        while True:
            # クライアント接続の認識
            client, addr = server.accept()
            data = bytes()
            if client:

                print('Connection received from %s:%d' % (addr[0], addr[1]))
                # client.send("")

                # クライアントのコネクションをハンドリングするスレッドの生成と実行
                client_handle_thread = threading.Thread(
                    target=self.client_handler,
                    args=(client,data)
                ).start()


    def client_handler(self,client,data):
        global addr
        buff_list = bytes()
        r_packet = bytes()

    # パケットの受信
        bytes_recd = 0
        MSGLEN = 4
        while True:
            tmp = client.recv(4096)
            if tmp ==b'':
                raise RuntimeError("socket connection broken")

            r_packet += tmp
            print(len(r_packet))
            if len(r_packet)>4:
                if len(r_packet) >= int.from_bytes(r_packet[0:4],'big'):
                    break
        r_packet = r_packet[4:]
        r_cmd = int.from_bytes(r_packet[0:2], 'big')
        print(r_cmd)
        # WAV保存の処理
        if r_cmd == WAV:
           # d_len = int.from_bytes(r_packet[6:8],'big')
           # data += r_cmd[4:d_len] 
            wav_id = int(time.time())
            output_path =  self.cla_dir[client.getpeername()[0]] +str(wav_id)+ ".wav"
            framerate = int.from_bytes(r_packet[2:6], 'big')
            samplewidth = int.from_bytes(r_packet[6:8], 'big')
            nchanneles = int.from_bytes(r_packet[8:10],'big')
            wf = wave.open(output_path, 'wb')
            wf.setnchannels(nchanneles)
            wf.setsampwidth(samplewidth)
            wf.setframerate(framerate)
            wf.writeframes(r_packet[10:])
            wf.close()
            text,type_ = speech_text(output_path)
            del r_packet
            gc.collect()
            print('テキスト化')
            print(text)
            pac = wav_id.to_bytes(5, 'big')
            if type_:
                pac += int(1).to_bytes(1,'big')
            else:
                pac += int(0).to_bytes(1,'big')
                
            text_b = text.encode()
            text_length = len(text_b)

            pac += text_length.to_bytes(2,'big')
            pac += text_b
            client.sendall(pac)
            print('sending text complete')
            client.close()
            
        # WAV再生の処理
        elif r_cmd == PLAY:
            wav_id = int.from_bytes(r_packet[2:], byteorder = "big")
            input_path =  self.cla_dir[client.getpeername()[0]]  + str(wav_id)+ ".wav"
            print(input_path)
            pac = bytes()
            waveFile = wave.open(input_path, 'r')
                    # wavファイルの情報を取得
            # チャネル数：monoなら1, stereoなら2, 5.1chなら6(たぶん)
            nchanneles = waveFile.getnchannels()

            # 音声データ1サンプルあたりのバイト数。2なら2bytes(16bit), 3なら24bitなど
            samplewidth = waveFile.getsampwidth()

            # サンプリング周波数。普通のCDなら44.1k
            framerate = waveFile.getframerate()

            # 音声のデータ点の数
            nframes = waveFile.getnframes()
            data = waveFile.readframes(-1)
            waveFile.close()
            pac = int(len(data)+12).to_bytes(4,'big')
            pac += framerate.to_bytes(4,'big')
            pac += samplewidth.to_bytes(2,'big')
            pac += nchanneles.to_bytes(2,'big')
            pac += data
            print('send wav')
            client.sendall(pac)
            client.close()
            
        # 要約の処理
        elif r_cmd == SUM:
            text = r_packet[2:].decode()
            print('summary from:',text)
            suma = ''
            if text:
                suma = Bertsum_pred(text)
            print('summary complete')
            suma = '\n'.join(suma)
            data = suma.encode()
            pac = int(len(data)+4).to_bytes(4,'big')
            pac += data
            print(len(data))
            client.sendall(pac)
            print('sended!')
            client.close()
            
        # スタートの処理
        elif r_cmd == SET:
            dir_path =  "./wav_file/" +str(int(time.time()))+"/"
            os.mkdir(dir_path)
            self.cla_dir[client.getpeername()[0]] = dir_path
            
        #wavfile受け取りの処理    
        elif r_cmd == INPUT:
            wav_id = int(time.time())
            framerate = int.from_bytes(r_packet[2:6], 'big')
            samplewidth = int.from_bytes(r_packet[6:8], 'big')
            nchanneles = int.from_bytes(r_packet[8:10],'big')
            
            if samplewidth == 2:
                data = np.frombuffer(r_packet[10:], dtype='int16')
            elif samplewidth == 4:
                data = np.frombuffer(r_packet[10:], dtype='int32')
                
            if nchanneles == 2:
                data = data[::nchanneles]
            data =[data[idx:idx + framerate] for idx in range(0,len(data), framerate)]
            save = 0
            save_data = bytes()
            pac = bytes()
            stop_counter = 0
            length = 0
            file_id = 0
            for d in data:
                
                # 閾値以上の場合はファイルに保存
                if d.max()/ 32768.0 > 0.1:
                    save = 1
                    
                if save == 1:
                    save_data += d.tobytes()
                    length += 1

                if d.max()/ 32768.0 <= 0.1 and save == 1:
                    stop_counter += 1
                    #設定秒間閾値を下回ったら一旦終了
                    if stop_counter >= 1:
                                               
                    #設定秒間以上だったら保存
                        if length  > 2:
                            Id = str(wav_id)+str(file_id)
                            print(Id)
                            file_path = self.cla_dir[client.getpeername()[0]] +Id+ ".wav"
                            
                            wf = wave.open(file_path, 'wb')
                            wf.setnchannels(nchanneles)
                            wf.setsampwidth(samplewidth)
                            wf.setframerate(framerate)
                            wf.writeframes(save_data)
                            wf.close()
                            print('save')
                            print(file_path)
                            
                            text,type_ = speech_text(file_path)
                            if text:
                                text += "。"
                            print('テキスト化')
                            print(text)
                            pac += int(Id).to_bytes(5, 'big')
                            if type_:
                                pac += int(1).to_bytes(1,'big')
                            else:
                                pac += int(0).to_bytes(1,'big')
                            text_b = text.encode()
                            text_length = len(text_b)
                            
                            pac += text_length.to_bytes(2,'big')
                            pac += text_b
       
                            file_id += 1
                            
                        stop_counter = 0    
                        length = 0
                        save = 0
                        save_data = bytes()
                        
            client.sendall(pac)
            print('sending text complete')
            print(len(pac))
            client.close()

        elif r_cmd == CON:
            data += r_packet[4:]
            self.client_handler(client,data)

        elif r_cmd == GIJI:
            texts = ""
            summary = ""
            tasks = ""
            t_len = int.from_bytes(r_packet[2:6], 'big')
            if t_len != 0:
                texts = r_packet[6:6+t_len].decode()
            s_len = int.from_bytes(r_packet[6+t_len:6+t_len+4],'big')
            if s_len != 0:
                summay = r_packet[6+t_len+4:6+t_len+4+s_len].decode()
          
            t_len = int.from_bytes(r_packet[6+t_len+4+s_len:6+t_len+4+s_len+4],'big')
            if t_len != 0:
                tasks = r_packet[6+t_len+4+s_len+4:].decode()

            gijiroku = { "texts":texts,"summasy":summay,"tasks":tasks}
     
            file_name ='./gijiroku/'+ str(int(time.time()))+'.json'
            with open(file_name,'w') as f:
                json.dump(gijiroku,f,ensure_ascii=False)

        else:
            print('a')
            client.close()

       # パケットをもどす
    def decode_packet(self,pac):
        
        #r_cmd = int.from_bytes(settings_list[0:1], 'big')
        FORMAT = int.from_bytes(pac[1:3],'big')
        CHANNELS = int.from_bytes(pac[3:5],'big')
        fs = int.from_bytes(pac[5:7],'big')
        RATE = int.from_bytes(pac[7:9],'big')
        CHUNK = int.from_bytes(pac[9:13],'big')

        return FORMAT,CHANNELS,fs,RATE,CHUNK


if __name__ == '__main__':
    port = 9012
    add="ip-172-31-40-14.ap-northeast-1.compute.internal"
    mss_server = StreamServer(add, port)
    mss_server.run()
