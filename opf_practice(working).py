# <20.07.24> by KH
import pandapower as pp
import pandapower.networks as pn
import numpy as np
from pandapower.pypower import idx_brch as ib



'''
Optimial Power Flow Class
'''
class OptimalPowerFlow:

    def __init__(self):
        self.max_loading_percent = 100
        self.max_i_ka = np.array([100, 100, 100, 60, 60, 60, 60, 60, 60, 60, 60])/230/np.sqrt(3)
        self.Z_BASE = (230**2)/100
        self.MAX_VM_PU = 1.07
        self.MIN_VM_PU = 0.95
        self.VN_KV = 230

        '''
        line_parameter[0]: r_ohm_per_km
        line_parameter[1]: x_ohm_per_km
        line_parameter[2]: c_nf_per_km
        '''
        self.line_parameter = np.array([(0.10,0.20,0.04), (0.05,0.20,0.04), (0.08,0.30,0.06), (0.05,0.25,0.06), (0.05,0.10,0.02), \
                                             (0.10,0.30,0.04), (0.07,0.20,0.05), (0.12,0.26,0.05), (0.02,0.10,0.02), (0.20,0.40,0.08), (0.10,0.30,0.06)])


    '''
    Create Network
    If you don't want default network, you have to fix this method
    '''
    def create_empty_network(self):
        self.net =pp.create_empty_network(name = "defualt_network", f_hz=60.0, sn_mva=100)

    def create_buses(self): # OPF 풀이 시 min_vm_pu, max_vm_pu는 반드시 필요함.
        self.bus1 = pp.create_bus(self.net, vn_kv=self.VN_KV, min_vm_pu=self.MIN_VM_PU, max_vm_pu=self.MAX_VM_PU, name="Reference_bus")
        self.bus2 = pp.create_bus(self.net, vn_kv=self.VN_KV, min_vm_pu=self.MIN_VM_PU, max_vm_pu=self.MAX_VM_PU)
        self.bus3 = pp.create_bus(self.net, vn_kv=self.VN_KV, min_vm_pu=self.MIN_VM_PU, max_vm_pu=self.MAX_VM_PU)
        self.bus4 = pp.create_bus(self.net, vn_kv=self.VN_KV, min_vm_pu=self.MIN_VM_PU, max_vm_pu=self.MAX_VM_PU)
        self.bus5 = pp.create_bus(self.net, vn_kv=self.VN_KV, min_vm_pu=self.MIN_VM_PU, max_vm_pu=self.MAX_VM_PU)
        self.bus6 = pp.create_bus(self.net, vn_kv=self.VN_KV, min_vm_pu=self.MIN_VM_PU, max_vm_pu=self.MAX_VM_PU)


    def create_lines(self):            # max_loading_percent 의 경우 OPF에서 반드시 설정해줘야함.
                # #+ max_i_ka: maximum thermal current in kilo Ampere 인데 정확하게 파악 필요
        # <20.07.27.> --check. c_fg_per_km=0.04 단위랑 다세히 다시 파악 필요.
        # real값을 구하기 때문에 Z_real을 구하기 위해서 Z_base 값 구함.
        # i_ka 값 계산 (flow_limit)/V

        pp.create_line_from_parameters(self.net, from_bus=self.bus1, to_bus=self.bus2, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[0][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[0][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[0][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[0], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus1, to_bus=self.bus4, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[1][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[1][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[1][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[1], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus1, to_bus=self.bus5, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[2][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[2][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[2][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[2], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus2, to_bus=self.bus3, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[3][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[3][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[3][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[3], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus2, to_bus=self.bus4, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[4][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[4][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[4][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[4], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus2, to_bus=self.bus5, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[5][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[5][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[5][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[5], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus2, to_bus=self.bus6, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[6][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[6][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[6][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[6], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus3, to_bus=self.bus5, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[7][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[7][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[7][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[7], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus3, to_bus=self.bus6, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[8][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[8][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[8][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[8], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus4, to_bus=self.bus5, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[9][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[9][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[9][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[9], max_loading_percent=self.max_loading_percent)

        pp.create_line_from_parameters(self.net, from_bus=self.bus5, to_bus=self.bus6, \
                                       length_km=1., r_ohm_per_km=self.line_parameter[10][0]*self.Z_BASE, x_ohm_per_km=self.line_parameter[10][1]*self.Z_BASE, \
                                       c_nf_per_km=1e9*self.line_parameter[10][2]/( self.Z_BASE*(2 *np.pi*self.net['f_hz']) ), max_i_ka= self.max_i_ka[10], max_loading_percent=self.max_loading_percent)

    def create_loads(self):
        pp.create_load(self.net, self.bus4, p_mw=100.0, q_mvar=15.0, controllable=False)
        pp.create_load(self.net, self.bus5, p_mw=100.0, q_mvar=15.0, controllable=False)
        pp.create_load(self.net, self.bus6, p_mw=100.0, q_mvar=15.0, controllable=False)

    def create_generators(self): # max/min_q_mvar/p_mw 4개는 OPF 풀이시 반드시 필요함.
        # vm_pu 이용해서 현재 전압 맞춰줄 필요가 있음.
        # <20.07.27.> sn_mva, type 추가 (type이 controllable과 관련있음. p_mw, q_mvar, vm_pu 가 강제됨)
        # <20.07.27.> vm_pu 의 set point 확인 필요.

        #min_vm_pu, max_vm_pur -> OPF 필수
        pp.create_gen(self.net, self.bus1, p_mw=110, vm_pu=1.07, sn_mva=100 ,min_p_mw=50.0, max_p_mw=200, \
                      min_q_mvar=-100.0, max_q_mvar=150.0, min_vm_pu=0.95, max_vm_pu=1.07, \
                      controllable=True, slack=True, type="controllable")

        pp.create_gen(self.net, self.bus2, p_mw=50, vm_pu=1.05, sn_mva=100,min_p_mw=37.5, max_p_mw=150, \
                      min_q_mvar=-100.0, max_q_mvar=150.0, min_vm_pu=0.95, max_vm_pu=1.07, \
                      controllable=True,  type="controllable")

        pp.create_gen(self.net, self.bus3, p_mw=50, vm_pu=1.05, sn_mva=100, min_p_mw=45.0, max_p_mw=180, \
                      min_q_mvar=-100.0, max_q_mvar=120.0, min_vm_pu=0.95, max_vm_pu=1.07, \
                      controllable=True,  type="controllable")

    '''
    Create Cost_function
    '''
    def create_cost_function(self): # <20.07.217.> --check. 무효전력 비용도 한번 고려해보기.
        cost_gen1 = pp.create_poly_cost(self.net, element=0, et="gen", \
                                        cp0_eur=213.1, cp1_eur_per_mw=11.669, \
                                        cp2_eur_per_mw2=0.00533)
        cost_gen2 = pp.create_poly_cost(self.net, element=1, et="gen", \
                                        cp0_eur=200.0, cp1_eur_per_mw=10.333, \
                                        cp2_eur_per_mw2=0.00889)
        cost_gen3 = pp.create_poly_cost(self.net, element=2, et="gen", \
                                        cp0_eur=240.0, cp1_eur_per_mw=10.833, \
                                        cp2_eur_per_mw2=0.00741)


        # net.poly_cost.cp1_eur_per_mw.at[costeg]=10
        # net.poly_cost.cp1_eur_per_mw.at[costgen1]=15
        # net.poly_cost.cp1_eur_per_mw.at[costgen2]=12

    '''
    Run Optimal Power Flow
    '''
    def run_opf(self): # calculate_volate_angles의 기본값이 False로 되어있음
        # init="pf" or "flat", pf 설정시 power flow 풀이 이후 초기값 설정. 시간이 오래걸림.
        pp.runopp(self.net, verbose=True, calculate_voltage_angles=True, init="pf")
        # pp.runpp(self.net, algorithm='nr', calculate_voltage_angles='auto', max_iteration=100)



    '''
    Show result
    '''
    def save_lagrange_multiplier_p(self):
        self.lam_p = self.net['res_bus']['lam_p'].sort_index() # 버스 별 lagrange multiplier 저장.

    def save_lagrange_multiplier_q(self):
        self.lam_q = self.net['res_bus']['lam_q'].sort_index()

    def iter_oper(self):
        self.create_empty_network()
        self.create_buses()
        self.create_lines()
        self.create_loads()
        self.create_generators()
        self.create_cost_function()
        self.run_opf()
        self.save_lagrange_multiplier_p()
        self.save_lagrange_multiplier_q()

if __name__ == "__main__":
    test_obj = OptimalPowerFlow()
    test_obj.iter_oper()

    print("=============================================\n", \
          "|    lambda P\n", \
          "=============================================\n", test_obj.lam_p,"\n",sep="")

    print("\n", test_obj.lam_q, "\n")
    print("=============================================\n", \
          "|    Generation result\n",
          "=============================================\n", \
          test_obj.net['res_gen'], "\n", sep="")
    print("|    total cost: ", test_obj.net.res_cost, "\n")

    print(test_obj.net['_ppc']['branch'][:, ib.MU_ST])
    print(test_obj.net['_ppc']['branch'][:, ib.MU_SF])