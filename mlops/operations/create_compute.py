import argparse
from utils import workspace, config
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

def create_compute_cluster(ws,
                   cpu_cluster_name,
                   vm_size='STANDARD_DS12_V2', 
                   min_nodes = 0,
                   max_nodes = 4,
                   idle_seconds_before_scaledown = 1800,
                   vnet_resourcegroup_name = None,
                   vnet_name=None, 
                   subnet_name=None):
    # Verify that cluster does not exist already
    try:
        cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
        print('Found existing cluster, use it.')
    except ComputeTargetException:
        # To use a different region for the compute, add a location='<region>' parameter
        compute_config = AmlCompute.provisioning_configuration(vm_size = vm_size, 
                                                            min_nodes = min_nodes,
                                                            max_nodes = max_nodes,
                                                            idle_seconds_before_scaledown = idle_seconds_before_scaledown,
                                                            vnet_resourcegroup_name = vnet_resourcegroup_name,
                                                            vnet_name = vnet_name, 
                                                            subnet_name = subnet_name,
                                                            remote_login_port_public_access = "Disabled")
                                                
        cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)


def main(name,
         vm_size='STANDARD_DS12_V2', 
         min_nodes = 0,
         max_nodes = 4,
         idle_seconds_before_scaledown = 1800,
         vnet_resourcegroup_name = None,
         vnet_name=None, 
         subnet_name=None):

    ws = workspace.retrieve_workspace()

    if not name:
        print("Compute target name not defined. Skipping.")
    if name in ws.compute_targets:
        print("Compute target already created. Skipping.")
    else:
        _ = create_compute_cluster(ws,
                   name,
                   vm_size, 
                   min_nodes,
                   max_nodes,
                   idle_seconds_before_scaledown,
                   vnet_resourcegroup_name,
                   vnet_name, 
                   subnet_name)


def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--vm_size", type=str)
    parser.add_argument("--min_nodes", type=int)
    parser.add_argument("--max_nodes", type=int)
    parser.add_argument("--idle_seconds_before_scaledown", type=int)
    parser.add_argument("--vnet_resourcegroup_name", type=str)
    parser.add_argument("--vnet_name", type=str)
    parser.add_argument("--subnet_name", type=str)

    return parser.parse_args(args_list)


if __name__ == "__main__":
    args = parse_args()
    name = args.name
    vm_size = args.vm_size
    min_nodes = args.min_nodes
    max_nodes = args.max_nodes
    idle_seconds_before_scaledown = args.idle_seconds_before_scaledown
    vnet_resourcegroup_name = args.vnet_resourcegroup_name
    vnet_name =args.vnet_name 
    subnet_name = args.subnet_name
    
    print(f"Initiating Compute Creation of {vm_size} with {min_nodes} min nodes and {max_nodes} max nodes")
    main(name,
         vm_size, 
         min_nodes,
         max_nodes,
         idle_seconds_before_scaledown,
         vnet_resourcegroup_name,
         vnet_name, 
         subnet_name)
