import math
import random
import numpy as np

class VECEnv(object):
    road = 400
    nECD = 2
    nVC = 100
    nCACHE = 4
    CACHE = 10
    nTasks = nVC
    nSubTask = 4
    B1 = 30
    B2 = 10
    noise = 10 ** (-9)
    r = 4
    A0 = 17.8 / 1000
    e2mt = 300
    e2mtr = 20
    TS = 4
    W1 = 0.5
    W2 = 0.5

    ECD_fecd = np.random.randint(120, 151, nECD)
    ECD_secd = np.random.randint(120, 151, nECD)
    ECD_pecd = np.random.randint(20, 25, nECD)
    ECD_tecd = np.random.randint(5, 10, nECD)
    ECD_lecd = [[100, 10], [300, 10]]
    ECD_sgecd = 200
    ECD_serecd = [[1, 2, 0, 4], [0, 2, 3, 4]]

    VC_fvc = np.random.randint(35, 41, nVC)
    VC_svc = np.random.randint(40, 46, nVC)
    VC_pvc = np.random.randint(15, 21, nVC)
    VC_tvc = np.random.randint(4, 8, nVC)
    X_coordinates = sorted(random.sample(range(1, road-4), nVC))
    VC_lvc = [[x, 0] for x in X_coordinates]
    VC_rvc = np.zeros(nVC, dtype=int)
    for i in range(nVC):
         VC_rvc[i] = int(X_coordinates[i] // ECD_sgecd)
    VC_spvc = np.random.randint(20, 25)
    VC_servc = np.random.randint(1, 5, nVC)

    subI = np.random.randint(5, 11, nVC)
    subO = np.random.randint(3, 7, nVC)
    subC = np.random.randint(10, 21, nVC)
    subS = np.random.randint(10, 16, nVC)
    subser = np.random.randint(1, 5, (nVC*nSubTask))
    DAG = np.random.randint(0, 2, nVC)
    Subtask_trans = np.ones(nTasks*nSubTask*(nECD+1)) * (1/3)

    def calculate_distance(point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        distance = np.sqrt(np.sum((point2 - point1) ** 2))
        return distance
    V2Edistances = []
    for vehicle in VC_lvc:
        distance1 = calculate_distance(vehicle, ECD_lecd[0])
        distance2 = calculate_distance(vehicle, ECD_lecd[1])
        V2Edistances.append([distance1, distance2])
    V2Egain = []
    for dist in V2Edistances:
        gain1 = 1 / (dist[0] ** r)
        gain2 = 1 / (dist[1] ** r)
        V2Egain.append([gain1, gain2])
    V2Etrans = []
    for i in range(len(VC_lvc)):
        v2etrans1 = B1 * math.log2(1 + VC_tvc[i] * V2Egain[i][0] / noise)
        v2etrans2 = B1 * math.log2(1 + VC_tvc[i] * V2Egain[i][1] / noise)
        V2Etrans.append([v2etrans1, v2etrans2])

    V2Vdistances = []
    V2Vgain = []
    V2Vtrans = []
    for i in range(len(VC_lvc)):
        row = []
        row1 = []
        row2 = []
        for j in range(len(VC_lvc)):
            distance = calculate_distance(VC_lvc[i], VC_lvc[j])
            if distance == 0:
                row.append(0)
                row1.append(0)
                row2.append(0)
            else:
                row.append(distance)
                row1.append(distance ** (-r))
                row2.append(B2 * math.log2(1 + VC_tvc[j] * row1[j] / (noise + A0 * row[j] ** (-r))))
        V2Vdistances.append(row)
        V2Vgain.append(row1)
        V2Vtrans.append(row2)

    action_space = [0, 1, 2]
    n_actions = len(action_space)
    n_states = 4 + 4
    def __init__(self):
        self.state = np.zeros((self.nVC, 4+4))
        for i in range(self.nVC):
            self.state1 = np.append(self.ECD_fecd, self.ECD_secd)
            self.state1 = np.append(self.state1, self.subI[i])
            self.state1 = np.append(self.state1, self.subO[i])
            self.state1 = np.append(self.state1, self.subC[i])
            self.state1 = np.append(self.state1, self.subS[i])
            self.state[i] = self.state1
    def reset(self):
        self.reset_env()
        self.state = np.zeros((self.nVC, 4+4))
        for i in range(self.nVC):
            self.state1 = np.append(self.ECD_fecd, self.ECD_secd)
            self.state1 = np.append(self.state1, self.subI[i])
            self.state1 = np.append(self.state1, self.subO[i])
            self.state1 = np.append(self.state1, self.subC[i])
            self.state1 = np.append(self.state1, self.subS[i])
            self.state[i] = self.state1
        return self._get_obs()
    def reset_env(self):
        self.ECD_fecd = np.random.randint(120, 151, self.nECD)
        self.ECD_secd = np.random.randint(120, 151, self.nECD)
        self.ECD_pecd = np.random.randint(20, 25, self.nECD)
        self.ECD_tecd = np.random.randint(5, 10, self.nECD)
        self.ECD_lecd = [[100, 10], [300, 10]]
        self.ECD_sgecd = 200
        self.ECD_serecd = [[1, 2, 0, 4], [0, 2, 3, 4]]

        self.VC_fvc = np.random.randint(35, 41, self.nVC)
        self.VC_svc = np.random.randint(40, 46, self.nVC)
        self.VC_pvc = np.random.randint(15, 21, self.nVC)
        self.VC_tvc = np.random.randint(4, 8, self.nVC)
        X_coordinates = sorted(random.sample(range(1, self.road-4), self.nVC))
        self.VC_lvc = [[x, 0] for x in X_coordinates]
        self.VC_rvc = np.zeros(self.nVC, dtype=int)
        for i in range(self.nVC):
            self.VC_rvc[i] = int(X_coordinates[i] // self.ECD_sgecd)
        self.VC_spvc = 5
        self.VC_servc = np.random.randint(1, 5, self.nVC)

        self.subI = np.random.randint(5, 11, self.nVC)
        self.subO = np.random.randint(3, 7, self.nVC)
        self.subC = np.random.randint(10, 21, self.nVC)
        self.subS = np.random.randint(10, 16, self.nVC)
        self.subser = np.random.randint(1, 5, (self.nVC * self.nSubTask))
        self.DAG = np.random.randint(0, 2, self.nVC)
    def _get_obs(self):
        self.state = np.zeros((self.nVC, 4 + 4))
        for i in range(self.nVC):
            self.state1 = np.append(self.ECD_fecd, self.ECD_secd)
            self.state1 = np.append(self.state1, self.subI[i])
            self.state1 = np.append(self.state1, self.subO[i])
            self.state1 = np.append(self.state1, self.subC[i])
            self.state1 = np.append(self.state1, self.subS[i])
            self.state[i] = self.state1
        return self.state

    def step(self, action, s, j, k, ac):
        s1 = np.array(s)
        if action == 0:
            s_ = np.append(s1[0:4], s1[4:]*0.9)
        elif action == 1:
            s_ = np.append(s1[0]-s1[6], s1[1])
            s_ = np.append(s_, s1[2]-s1[7])
            s_ = np.append(s_, s[3])
            s_ = np.append(s_, s1[4:]*0.9)
        elif action == 2:
            s_ = np.append(s1[0], s1[1]-s1[6])
            s_ = np.append(s_, s1[2])
            s_ = np.append(s_, s[3]-s1[7])
            s_ = np.append(s_, s1[4:] * 0.9)

        if j == 3:
            done = True
        else:
            done = False
        r, Tsub, Esub = self.compution(action, s_, j, k, ac)
        return s_, r, done, Tsub, Esub

    def compution(self, action, s_, j, k, ac):
        b = int(self.VC_rvc[k])
        c = s_[4] / self.V2Etrans[k][b] + s_[6] / self.ECD_fecd[b]
        if c * self.VC_spvc <= self.ECD_lecd[b][0] + self.ECD_sgecd / 2 - self.VC_lvc[k][0]:
            vcrate = 1
        else:
            vcrate = 0
        W1 = self.W1
        W2 = self.W2

        if self.DAG[k] == 0:
            if j == 0:
                if action == 0 and self.subser[k*self.nSubTask+j] == self.VC_servc[k]:
                    Tsub = s_[6] / self.VC_fvc[k]
                    Esub = self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]]:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]]:
                    Tsub = self.CACHE / self.e2mt + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.e2mtr * self.CACHE / self.e2mt + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and vcrate == 1:
                    Tsub = s_[4] / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[4] / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and vcrate == 0:
                    cc = self.VC_rvc[k] + 1
                    aa = np.argmax(self.VC_rvc == cc)
                    Tsub = s_[4] / self.V2Vtrans[k][aa] + s_[4] / self.V2Etrans[aa][self.VC_rvc[aa]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[4] / self.V2Vtrans[k][aa] + self.VC_tvc[aa] * s_[4] / self.V2Etrans[aa][self.VC_rvc[aa]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and vcrate == 1:
                    Tsub = s_[4] / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[4] / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and vcrate == 0:
                    cc = self.VC_rvc[k] + 1
                    aa = np.argmax(self.VC_rvc == cc)
                    Tsub = s_[4] / self.V2Vtrans[k][aa] + s_[4] / self.V2Etrans[aa][self.VC_rvc[aa]] + self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[4] / self.V2Vtrans[k][aa] + self.VC_tvc[aa] * s_[4] / self.V2Etrans[aa][self.VC_rvc[aa]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]

            if j == 1 or j == 2:
                if action == 0 and self.subser[k*self.nSubTask+j] == self.VC_servc[k] and ac[k, j-1] == 0:
                    Tsub = s_[6] / self.VC_fvc[k]
                    Esub = self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] == self.VC_servc[k] and ac[k, j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] == 0:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] == 0:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] == 0 and vcrate == 1:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] == 0 and vcrate == 0:
                    cc = self.VC_rvc[k] + 1
                    aa = np.argmax(self.VC_rvc == cc)
                    Tsub = s_[5]/0.9 / self.V2Vtrans[k][aa] + s_[5]/0.9 / self.V2Etrans[aa][self.VC_rvc[aa]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Vtrans[k][aa] + self.VC_tvc[aa] * s_[5]/0.9 / self.V2Etrans[aa][self.VC_rvc[aa]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] != 0:
                    Tsub = s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] == 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] != 0:
                    Tsub = self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]

            if j == 3:
                if action == 0 and self.subser[k*self.nSubTask+j] == self.VC_servc[k] and ac[k, j-1] == 0:
                    Tsub = s_[6] / self.VC_fvc[k]
                    Esub = self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] == self.VC_servc[k] and ac[k, j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] == 0:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] == 0:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] == 0 and vcrate == 1:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] == 0 and vcrate == 0:
                    cc = self.VC_rvc[k] + 1
                    aa = np.argmax(self.VC_rvc == cc)
                    Tsub = s_[5]/0.9 / self.V2Vtrans[k][aa] + s_[5]/0.9 / self.V2Etrans[aa][self.VC_rvc[aa]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Vtrans[k][aa] + self.VC_tvc[aa] * s_[5]/0.9 / self.V2Etrans[aa][self.VC_rvc[aa]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] != 0:
                    Tsub = s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] == 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] != 0:
                    Tsub = self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
        else:
            if j ==0:
                if action == 0 and self.subser[k * self.nSubTask + j] == self.VC_servc[k]:
                    Tsub = s_[6] / self.VC_fvc[k]
                    Esub = self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k * self.nSubTask + j] != self.VC_servc[k] and self.subser[k * self.nSubTask + j] in self.ECD_serecd[self.VC_rvc[k]]:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k * self.nSubTask + j] != self.VC_servc[k] and self.subser[k * self.nSubTask + j] not in self.ECD_serecd[self.VC_rvc[k]]:
                    Tsub = self.CACHE / self.e2mt + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.e2mtr * self.CACHE / self.e2mt + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action != 0 and self.subser[k * self.nSubTask + j] in self.ECD_serecd[self.VC_rvc[k]] and vcrate == 1:
                    Tsub = s_[4] / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[4] / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k * self.nSubTask + j] in self.ECD_serecd[self.VC_rvc[k]] and vcrate == 0:
                    cc = self.VC_rvc[k] + 1
                    aa = np.argmax(self.VC_rvc == cc)
                    Tsub = s_[4] / self.V2Vtrans[k][aa] + s_[4] / self.V2Etrans[aa][self.VC_rvc[aa]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[4] / self.V2Vtrans[k][aa] + self.VC_tvc[aa] * s_[4] / self.V2Etrans[aa][self.VC_rvc[aa]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k * self.nSubTask + j] not in self.ECD_serecd[self.VC_rvc[k]] and vcrate == 1:
                    Tsub = s_[4] / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[4] / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k * self.nSubTask + j] not in self.ECD_serecd[self.VC_rvc[k]] and vcrate == 0:
                    cc = self.VC_rvc[k] + 1
                    aa = np.argmax(self.VC_rvc == cc)
                    Tsub = s_[4] / self.V2Vtrans[k][aa] + s_[4] / self.V2Etrans[aa][self.VC_rvc[aa]] + self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[4] / self.V2Vtrans[k][aa] + self.VC_tvc[aa] * s_[4] / self.V2Etrans[aa][self.VC_rvc[aa]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]

            if j == 1 or j == 2:
                if action == 0 and self.subser[k * self.nSubTask + j] == self.VC_servc[k] and ac[k, j//self.nSubTask] == 0:
                    Tsub = s_[6] / self.VC_fvc[k]
                    Esub = self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k * self.nSubTask + j] == self.VC_servc[k] and ac[k, j//self.nSubTask] != 0:
                    Tsub = s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k * self.nSubTask + j] != self.VC_servc[k] and self.subser[k * self.nSubTask + j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] == 0:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k * self.nSubTask + j] != self.VC_servc[k] and self.subser[k * self.nSubTask + j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] != 0:
                    Tsub = s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k * self.nSubTask + j] != self.VC_servc[k] and self.subser[k * self.nSubTask + j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] == 0:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k * self.nSubTask + j] != self.VC_servc[k] and self.subser[k * self.nSubTask + j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] != 0:
                    Tsub = s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action != 0 and self.subser[k * self.nSubTask + j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] == 0 and vcrate == 1:
                    Tsub = s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k * self.nSubTask + j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] == 0 and vcrate == 0:
                    cc = self.VC_rvc[k] + 1
                    aa = np.argmax(self.VC_rvc == cc)
                    Tsub = s_[5] / 0.9 / self.V2Vtrans[k][aa] + s_[5] / 0.9 / self.V2Etrans[aa][self.VC_rvc[aa]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5] / 0.9 / self.V2Vtrans[k][aa] + self.VC_tvc[aa] * s_[5] / 0.9 / self.V2Etrans[aa][self.VC_rvc[aa]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k * self.nSubTask + j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] != 0:
                    Tsub = s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k * self.nSubTask + j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] == 0:
                    Tsub = s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5] / 0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                if action != 0 and self.subser[k * self.nSubTask + j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j//self.nSubTask] != 0:
                    Tsub = self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]]
                    Esub = self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]]
            if j == 3:
                if action == 0 and self.subser[k*self.nSubTask+j] == self.VC_servc[k] and ac[k, j-1] == 0:
                    Tsub = s_[6] / self.VC_fvc[k]
                    Esub = self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] == self.VC_servc[k] and ac[k, j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] == 0:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k, j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] == 0:
                    Tsub = self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action == 0 and self.subser[k*self.nSubTask+j] != self.VC_servc[k] and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] != 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.VC_fvc[k]
                    Esub = self.ECD_tecd[self.VC_rvc[k]] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_tecd[self.VC_rvc[k]] * self.CACHE / self.V2Etrans[k][self.VC_rvc[k]] + self.VC_pvc[k] * s_[6] / self.VC_fvc[k]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] == 0 and vcrate == 1:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] == 0 and vcrate == 0:
                    cc = self.VC_rvc[k] + 1
                    aa = np.argmax(self.VC_rvc == cc)
                    Tsub = s_[5]/0.9 / self.V2Vtrans[k][aa] + s_[5]/0.9 / self.V2Etrans[aa][self.VC_rvc[aa]] + s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Vtrans[k][aa] + self.VC_tvc[aa] * s_[5]/0.9 / self.V2Etrans[aa][self.VC_rvc[aa]] + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] != 0:
                    Tsub = s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] == 0:
                    Tsub = s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.VC_tvc[k] * s_[5]/0.9 / self.V2Etrans[k][self.VC_rvc[k]] + self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                if action != 0 and self.subser[k*self.nSubTask+j] not in self.ECD_serecd[self.VC_rvc[k]] and ac[k,j-1] != 0:
                    Tsub = self.CACHE / self.e2mt + s_[6] / self.ECD_fecd[self.VC_rvc[k]] + s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
                    Esub = self.e2mtr * self.CACHE / self.e2mt + self.ECD_pecd[self.VC_rvc[k]] * s_[6] / self.ECD_fecd[self.VC_rvc[k]] + self.ECD_tecd[self.VC_rvc[k]] * s_[5] / self.V2Etrans[k][self.VC_rvc[k]]
        r = -(W1 * Tsub + W2 * Esub)
        return r, Tsub, Esub