'''
conda install -c conda-forge pyomo pyomo.extra
conda install -c conda-forge ipopt glpk
'''

'''
# <20.08.02> by KH
The issue is that if I run this optimization problem,

It is not working because of object function's data type problem
(It looks like object functions are int type now)

before I start coding something, I should check the data type of 'Param' and 'Set'

and also should check the how does it work

after then maybe I can handle the indexed constraints, variables

# <20.08.03> by KH

result reference 
<ref = https://coin-or.github.io/Ipopt/OUTPUT.html >


# <20.08.03> by KH
최적화 자코비안 부분 남음.
해당 부분 해결을 위해서 전체를 표현할 수 있는 변수를 선언할 필요가 있음

'''

from pyomo.environ import *
import pyomo.contrib.parmest.parmest as pamest
import pickle
import numpy as np

class OPF_Optimization:
    def __init__(self, file_name):
        self.file_name = file_name
        self.bus_param = []
        self.line_param = []
        self.load_param = []
        self.gen_param = []
        self.cost_param= []
        
        self.bus_param_table = []
        self.line_param_table = []
        self.load_param_table = []
        self.gen_param_table = []
        self.cost_param_table = []

        self.buffer = []

    def get_network_model(self):
        with open(self.file_name,"rb") as fr:
            self.network = pickle.load(fr)

    def set_model(self):
        self.model = ConcreteModel()  

    def take_number(self):
        self.NUM_BUS = len(self.network['bus'])
        self.NUM_LINE = len(self.network['line'])
        self.NUM_GEN = len(self.network['gen'])
        self.NUM_LOAD = len(self.network['load'])

        self.NUM_BUS_PARAM = len(self.network['bus'].columns)
        self.NUM_LINE_PARAM = len(self.network['line'].columns)
        self.NUM_GEN_PARAM = len(self.network['gen'].columns)
        self.NUM_LOAD_PARAM = len(self.network['load'].columns)

    def set_bus_parameter(self):
        for bus_column in range(self.NUM_BUS_PARAM):
            for bus_nu in range(self.NUM_BUS):
                self.buffer.append(self.network['bus'][self.network['bus'].columns[bus_column]][bus_nu])

            self.bus_param.append(self.buffer.copy())
            self.buffer.clear()

            self.bus_param_table.append(self.network['bus'].columns[bus_column])

    def set_line_parameter(self):
        for line_column in range(self.NUM_LINE_PARAM):
            for line_nu in range(self.NUM_LINE):
                self.buffer.append(self.network['line'][self.network['line'].columns[line_column]][line_nu])

            self.line_param.append(self.buffer.copy())
            self.buffer.clear()

            self.line_param_table.append(self.network['line'].columns[line_column])

    def set_load_parameter(self):
        for load_column in range(self.NUM_LOAD_PARAM):
            for load_nu in range(self.NUM_LOAD):
                self.buffer.append(self.network['load'][self.network['load'].columns[load_column]][load_nu])

            self.load_param.append(self.buffer.copy())
            self.buffer.clear()

            self.load_param_table.append(self.network['load'].columns[load_column])

    def set_gen_parameter(self):
        for gen_column in range(self.NUM_GEN_PARAM):
            for gen_nu in range(self.NUM_GEN):
                self.buffer.append(self.network['gen'][self.network['gen'].columns[gen_column]][gen_nu])

            self.gen_param.append(self.buffer.copy())
            self.buffer.clear()

            self.gen_param_table.append(self.network['gen'].columns[gen_column])

    def set_cost_parameter(self):
        if len(self.network['pwl_cost'])==0:
            print("************************ loading poly cost function ************************")
            self.NUM_COST_PARAM = len(self.network['poly_cost'].columns)
            
            for cost_column in range(self.NUM_COST_PARAM):
                for gen_nu in range(self.NUM_GEN):
                    self.buffer.append(self.network['poly_cost'][self.network['poly_cost'].columns[cost_column]][gen_nu])
                
                self.cost_param.append(self.buffer.copy())
                self.buffer.clear()

                self.cost_param_table.append(self.network['poly_cost'].columns[cost_column])


    def set_y_bus(self):
        try:
            self.y_matrix_array = self.network['_ppc']['internal']['Ybus'].toarray()
		
        except:
            print("Y_bus import error")

    def set_jacobian(self):
        # <20.08.03.> Jacobian matrix reference
        # <ref = https://www.gitmemory.com/issue/e2nIEE/pandapower/381/498016844 >
        try:
            self.jacobian_array = self.network['_ppc']['internal']['J'].toarray()


        except:
            print("Jacobian matrix import error")


    '''
    Optimization Part
    '''
    
    def set_parameter(self):
        # find generation power index
        #self.generation_power_index = self.gen_param_table.index('p_mw')   # --check. 현재 발전량 어디에 쓸지 파악해보기.
        self.maximum_generation_power_index = self.gen_param_table.index('max_p_mw')
        self.minimum_generation_power_index = self.gen_param_table.index('min_p_mw')

        # set generation power from power flow result
        self.generation_power_0_mw = self.network['res_gen']['p_mw']
        self.generation_power_0_mva = self.network['res_gen']['q_mvar']


        # find power generation cost
        self.cp0_cost = self.cost_param_table.index('cp0_eur')		# --check. I have to stick "_index" at the end of the variable name
        self.cp1_cost = self.cost_param_table.index('cp1_eur_per_mw')
        self.cp2_cost = self.cost_param_table.index('cp2_eur_per_mw2')
        self.cq0_cost = self.cost_param_table.index('cq0_eur')
        self.cq1_cost = self.cost_param_table.index('cq1_eur_per_mvar')
        self.cq2_cost = self.cost_param_table.index('cq2_eur_per_mvar2')

        # find bus voltage, index parameter
        self.bus_voltage_max_index = self.bus_param_table.index('max_vm_pu')
        self.bus_voltage_min_index = self.bus_param_table.index('min_vm_pu')

        # voltage magnitude from power flow
        self.bus_voltage_magnitude_0 = self.network['res_bus']['vm_pu']
        self.bus_voltage_angle_ = self.network['res_bus']['va_degree']

        # find load active, reactive power
        self.load_MW_power_index = self.load_param_table.index('p_mw')
        self.load_MVA_power_index = self.load_param_table.index('q_mvar')

        # find line max flow index parameter
        self.maximum_line_flow_limit_index = [100 for i in range(3)] + [ 60 for j in range(8)] # <20.08.03.> --check. I have to notice this at the top. and fix it later
        self.from_bus_index = self.line_param_table.index('from_bus')
        self.to_bus_index = self.line_param_table.index('to_bus')
        #self.maximum_line_react_flow_limit_index = self.line_param_table.index('')	# At panda power they handle line limits with max_i_ka, max_loading_percent

    def set_variable(self):
        # set generation power variable
        self.model.delta_generation_power_mw = Var(range(self.NUM_GEN))  # <20.08.03> --check. I have to consider initialize generation power
        self.model.delta_generation_power_mva = Var(range(self.NUM_GEN))  # <20.08.03> --check. I have to consider initialize generation power

        # set generation cost function variable
        self.model.generation_cost = Var(range(self.NUM_GEN))

        # set bus voltage, angle variable
        self.model.delta_bus_voltage = Var(range(self.NUM_BUS))
        # self.model.bus_angle = Var(range(self.NUM_BUS))       # <20.08.03> --check. have to consider later

    def set_constraints(self):
        self.model.limits = ConstraintList()
        for i in self.model.delta_generation_power_mw: # Important expression, generation power constraint --check. also have to make Q constraint

            # upper and lower limits on the generator real and reactive power
            self.model.limits.add(self.model.delta_generation_power_mw[i]<= self.gen_param[self.maximum_generation_power_index][i] - self.generation_power_0_mw[i])
            self.model.limits.add(self.model.delta_generation_power_mw[i]>= self.gen_param[self.minimum_generation_power_index][i] - self.generation_power_0_mw[i])

            self.model.limits.add(self.model.delta_generation_power_mva[i]<= self.gen_param[self.maximum_generation_power_index][i] - self.generation_power_0_mva[i])
            self.model.limits.add(self.model.delta_generation_power_mva[i]>= self.gen_param[self.minimum_generation_power_index][i] - self.generation_power_0_mva[i])


            # jacobian matrix
            if self.bus_param[0][i] != 'reference_bus':
                pass
            else:
                pass



        for j in self.model.generation_cost: # Generation Cost Function define
            self.model.limits.add(self.model.generation_cost[j] == (self.cost_param[self.cp0_cost][j] ) \
                                                                    + (self.cost_param[self.cp1_cost][j] * self.generation_power_0_mw[j]) \
                                                                    + (self.cost_param[self.cp2_cost][j] * (self.generation_power_0_mw[j])**2) \
                                                                    + ((self.cost_param[self.cp1_cost][j]+ 2*(self.cost_param[self.cp2_cost][j])) * self.model.delta_generation_power_mw[j]) )


        for k in self.model.delta_bus_voltage: # Voltage Magnitude constraint
            self.model.limits.add(self.model.delta_bus_voltage[k] <= self.bus_param[self.bus_voltage_max_index][k] - self.bus_voltage_magnitude_0[k])
            self.model.limits.add(self.model.delta_bus_voltage[k] >= self.bus_param[self.bus_voltage_min_index][k] - self.bus_voltage_magnitude_0[k])


 
    def set_object_function(self):
        self.model.obj = Objective( expr = summation(self.model.generation_cost), sense=minimize)

    def set_solver(self):
        self.solver = SolverFactory("ipopt")
        # self.solver = SolverFactory("BONMIN")

    def solve(self):
        result = self.solver.solve(self.model, tee=True)
        return result


if __name__ == "__main__":
    test_model = OPF_Optimization("network_data")
    test_model.get_network_model()
    test_model.take_number()
    test_model.set_bus_parameter()
    test_model.set_line_parameter()
    test_model.set_load_parameter()
    test_model.set_gen_parameter()
    test_model.set_cost_parameter()
    test_model.set_y_bus()
    test_model.set_jacobian()
    
    test_model.set_model()
    test_model.set_parameter()
    test_model.set_variable()
    test_model.set_constraints()
    test_model.set_object_function()
    test_model.set_solver()
    test_model.solve()

    print("==========================================================")
    print("|     결정 변수 값")
    print("==========================================================\n")
    for v in test_model.model.component_objects(Var, active=True):
        print("Variable", v)

        for index in v:
            print("    ", index, value(v[index]))

    print("==========================================================")
    print("|     실제 발전량")
    print("==========================================================\n")
    for i in test_model.model.delta_generation_power_mw:
        print(i, "\t", test_model.generation_power_0_mw[i] + value(test_model.model.delta_generation_power_mw[i]))
