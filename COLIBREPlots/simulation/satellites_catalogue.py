import numpy as np

def look_for_satellites(sim_info, sample_host):

    host_index = []
    satellites_index = []
    satellites_sample = []
    num_sample = len(sample_host)

    for i in range(num_sample):

        x = sim_info.halo_data.xminpot - sim_info.halo_data.xminpot[sample_host[i]]
        y = sim_info.halo_data.yminpot - sim_info.halo_data.yminpot[sample_host[i]]
        z = sim_info.halo_data.zminpot - sim_info.halo_data.zminpot[sample_host[i]]

        distance = np.sqrt(x**2 + y**2 + z**2)
        Rvir = sim_info.halo_data.virial_radius[sample_host[i]]

        select = np.where((distance>0) & (distance<= 2 * Rvir))[0]
        flag = sim_info.halo_data.satellite_flag[select]
        sub_select = np.where(flag == True)[0]
        select = select[sub_select]

        sub_select = np.where(sim_info.halo_data.log10_halo_mass[select] > 7)[0]
        select = select[sub_select]
        num_satellites = len(select)

        satellites_sample = np.append(satellites_sample, select)
        satellites_index = np.append(satellites_index, sim_info.halo_data.halo_index[select])
        host_index = np.append(host_index, np.ones(num_satellites) * sim_info.halo_data.halo_index[sample_host[i]] )

    satellites_sample = satellites_sample.astype('int')

    return satellites_sample, satellites_index, host_index





