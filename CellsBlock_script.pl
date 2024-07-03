#!/usr/bin/perl
use strict;
use warnings FATAL => 'all';
# use Carp;
use Carp;
use feature qw(switch);
# no warnings 'experimental::smartmatch';
use Storable qw(dclone);

use Verilog::Netlist;
#use Data::Dumper;

# Setup options so files can be found
use Verilog::Getopt;

use Time::HiRes qw( time );
use List::Util qw(first);

use POSIX qw(localtime asctime);
use JSON qw(encode_json decode_json);
use File::Path qw(make_path);


use FindBin qw($Bin);
use lib "$Bin";
# use NetRD_TransProb_func qw(TransProb_caluer);
# use feature_functions qw(cellsblock_finder logic_loop_finder logic_SSC_finder sensitive_path_finder DominatorTree_builder);
use Getopt::Long;

my $start = time();


my $design_folder_dir='';
my $type_folder_dir='';
my $source_file_name='';

my $ref_Gate_HASH_ORI_fname     = '';
my $ref_Netlink_HASH_fname      = '';
my $ref_Design_input_Port_fname = '';
my $ref_Design_output_Port_fname= '';
my $logic_level = 3;

GetOptions ( "design_folder_path=s"      => \$design_folder_dir, # string
             "subtype_folder_path=s"     => \$type_folder_dir, # string
             "source_name=s"             => \$source_file_name,
             "ref_Gate_HASH_ORI=s"       => \$ref_Gate_HASH_ORI_fname,
             "ref_Netlink_HASH=s"        => \$ref_Netlink_HASH_fname,
             "ref_Design_input_Port=s"   => \$ref_Design_input_Port_fname,
             "ref_Design_output_Port=s"  => \$ref_Design_output_Port_fname,
             "logic_level=i"             => \$logic_level)
    or die("Error in command line arguments\n");


my $base_dir=$Bin;
my $sub_dir='json_temp_file';
my $sub_path = $base_dir.'/'.$sub_dir;
my $sub_design_folder_path = $sub_path.'/'.$design_folder_dir;
my $sub_subtype_folder_path = $sub_design_folder_path.'/'.$type_folder_dir;
my $sub_source_file_folder_path = $sub_subtype_folder_path.'/'.$source_file_name;


my %loaded_hash=( 'ref_Gate_HASH_ORI'=>$ref_Gate_HASH_ORI_fname,
                  'ref_Netlink_HASH'=>$ref_Netlink_HASH_fname,
                  'ref_Design_input_Port'=>$ref_Design_input_Port_fname,
                  'ref_Design_output_Port'=>$ref_Design_output_Port_fname
                );

foreach my $var (keys %loaded_hash){
    my $file_name=$loaded_hash{$var};
    my $filter_log=$sub_source_file_folder_path.'/'.$file_name;
    open(my $Log_Handle, '<',$filter_log) or die "Log File can not be open, $!";
    $loaded_hash{$var}='';
    while(<$Log_Handle>){
        $loaded_hash{$var} .= $_;
    }
    close($Log_Handle);
    $loaded_hash{$var}=decode_json($loaded_hash{$var});

}

printf("####################\n");
printf("Feature Stage: \n");
printf("cell_block in logic distance\n");
printf("####################\n");

my $sub1_filter_log=$sub_source_file_folder_path.'/'.'cell_block_extract.txt';
open(my $sub1_Log_Handle, '>',$sub1_filter_log) or die "cell_block_extract.txt File can not be open, $!";

my %hash_blocks=();
foreach my $Gate (sort (keys%{$loaded_hash{'ref_Gate_HASH_ORI'}})){
    printf($sub1_Log_Handle "*cell: %s\n", $Gate);
    my $ref_hash_cell_block=cellsblock_finder( $loaded_hash{'ref_Gate_HASH_ORI'},
                                            $loaded_hash{'ref_Netlink_HASH'},
                                            $loaded_hash{'ref_Design_input_Port'},
                                            $loaded_hash{'ref_Design_output_Port'},
                                            $Gate,
                                            $logic_level);
    $hash_blocks{$Gate}=$ref_hash_cell_block;
    foreach my $side (sort (keys %{$ref_hash_cell_block})){
        printf($sub1_Log_Handle "**Side: %s\n", $side);
        for my $cell (@{$$ref_hash_cell_block{$side}}){
            if(defined $$cell[7]){
                printf($sub1_Log_Handle "%s", "   "x($logic_level-$$cell[7]));
            }
            for my $ele (@{$cell}){
                printf($sub1_Log_Handle "%s, ",$ele);
            }
            printf($sub1_Log_Handle "\n");
        }

    }

}

close $sub1_Log_Handle;


my %return_hash=( 'ref_hash_blocks'=>\%hash_blocks,
                );

foreach my $var (keys %return_hash){
    my $json_data = encode_json($return_hash{$var});
    my $file_name = $var.'.json';
    my $filter_log=$sub_source_file_folder_path.'/'.$file_name;
    open(my $Log_Handle, '>',$filter_log) or die "Log File can not be open, $!";
    print $Log_Handle $json_data;
    close $Log_Handle;
}
my $end = time();

printf("used time: %fs\n", $end - $start);





















printf("####################\n");

sub cellsblock_finder{
    my $ref_gate_hash=shift;
    my $ref_netlink_hash=shift;
    my $ref_design_input_port=shift;
    my $ref_design_output_port=shift;
    my $cell_local_name= shift;
    ####################################
    my $Logic_dist=shift;
    ###################################
    printf("(cellsblock_finder): current cell: %s\n",$cell_local_name);
    ###################################
    my %return_blk_hash=();
    #--input side
    my @block_list_L=func_cells_finder($cell_local_name, $ref_gate_hash, $ref_netlink_hash, $ref_design_input_port, $ref_design_output_port, $Logic_dist, "L"); #"L" OR "R" which is the searching direction of cells_finder
    $return_blk_hash{'L'}=\@block_list_L;
    #--output side
    my @block_list_R=func_cells_finder($cell_local_name, $ref_gate_hash, $ref_netlink_hash, $ref_design_input_port, $ref_design_output_port, $Logic_dist, "R");
    $return_blk_hash{'R'}=\@block_list_R;

    return \%return_blk_hash;

}

sub func_cells_finder{
    my $current_cell_name=shift;
    my $ref_gate_hash=shift;
    my $ref_netlink_hash=shift;
    my $ref_design_input_port=shift;
    my $ref_design_output_port=shift;
    my $cur_logic_dist=shift;
    my $direction=shift;

    # my @matched_cell_name_list=();
    my @matched_cell_list=();
    if($cur_logic_dist == 0 ){
    }
    else{
#         # 1.save current cell to list
#         printf("current cell: %s\n",$current_cell_name);
#         # 2.get the all the prev_cells and forward_cells to a list
        if($direction eq "L"){
            # my $start = time();
            my $current_cell_refname =   $$ref_gate_hash{$current_cell_name}{'cell_reference_name'};
            my @current_net_list     = @{$$ref_gate_hash{$current_cell_name}{'list_input_net'}};
            my @current_pin_list     = @{$$ref_gate_hash{$current_cell_name}{'list_input_pin'}};
            my $index_a=0;
            foreach my $ref_net (@current_net_list){
                if(exists $$ref_netlink_hash{$ref_net}) {
                    my %linked_net = %{$$ref_netlink_hash{$ref_net}};
                    # $net{'list_input_cell_pin'}
                    # $net{'no_inputs'}
                    # $net{'list_output_cell_pin'}
                    # $net{'no_outputs'}
                    if(@{$linked_net{'list_output_cell_pin'}}){
                        foreach my $cell_pin (@{$linked_net{'list_output_cell_pin'}}) {
                            my @matched_cell_element = ();
                            # printf("%s,", @{$cell_pin}[0]);
                            # printf("%s,", @{$cell_pin}[1]);
                            # printf("%s\n", @{$cell_pin}[2]);
                            # matched_cell_element: dst_cell_name, dst_cell_refname, dst_cell_pin_name, net,
                            # cur_cell_name, cur_cell_refname, cur_cell_pin_name,logic_dist
                            @matched_cell_element = (@{$cell_pin}[0], @{$cell_pin}[1], @{$cell_pin}[2], $ref_net,
                                $current_cell_name, $current_cell_refname, $current_pin_list[$index_a], $cur_logic_dist);
                            push(@matched_cell_list, \@matched_cell_element);

                            my @rest_matched_cell_list = func_cells_finder(@{$cell_pin}[0], $ref_gate_hash, $ref_netlink_hash,$ref_design_input_port, $ref_design_output_port, $cur_logic_dist - 1, $direction);
                            push(@matched_cell_list, @rest_matched_cell_list);
                        }
                    }else{#$ref_design_input_port
                         if(grep /^$ref_net$/, @{$ref_design_input_port}) {
                            my @matched_cell_element = ();
                            @matched_cell_element=("input", "input", "input",$ref_net,
                                                    $current_cell_name,$current_cell_refname, $current_pin_list[$index_a], $cur_logic_dist);
                            push(@matched_cell_list,\@matched_cell_element);
                         }else{
                             my @matched_cell_element = ();
                             @matched_cell_element=("None_I", "None_I", "None_I",$ref_net,
                                                    $current_cell_name,$current_cell_refname, $current_pin_list[$index_a], $cur_logic_dist);
                             push(@matched_cell_list,\@matched_cell_element);
                         }
                    }

                }else{
                    croak ("$ref_net was not found in ref_netlink_hash \n");
                }
                $index_a=$index_a+1;
            }

        }
        elsif($direction eq "R"){
            my $current_cell_refname    =  $$ref_gate_hash{$current_cell_name}{'cell_reference_name'};
            my @current_net_list        =@{$$ref_gate_hash{$current_cell_name}{'list_output_net'}};
            my @current_pin_list        =@{$$ref_gate_hash{$current_cell_name}{'list_output_pin'}};
            my $index_a=0;
            foreach my $ref_net (@current_net_list){
                if(exists $$ref_netlink_hash{$ref_net}){
                     my %linked_net=%{$$ref_netlink_hash{$ref_net}};
                    if(@{$linked_net{'list_input_cell_pin'}}){# list_input_cell_pin is not empty
                        foreach my $cell_pin (@{$linked_net{'list_input_cell_pin'}}){
                            my @matched_cell_element=();
                            @matched_cell_element=(@{$cell_pin}[0], @{$cell_pin}[1], @{$cell_pin}[2],$ref_net,
                                                    $current_cell_name,$current_cell_refname, $current_pin_list[$index_a], $cur_logic_dist);
                            push(@matched_cell_list,\@matched_cell_element);

                            my @rest_matched_cell_list=func_cells_finder(@{$cell_pin}[0],$ref_gate_hash,$ref_netlink_hash,$ref_design_input_port, $ref_design_output_port, $cur_logic_dist-1,$direction);
                            push(@matched_cell_list,@rest_matched_cell_list);
                        }
                    }else{#$ref_design_output_port
                        if(grep /^$ref_net$/, @{$ref_design_output_port}) {
                            my @matched_cell_element = ();
                            @matched_cell_element=("output", "output", "output",$ref_net,
                                                    $current_cell_name,$current_cell_refname, $current_pin_list[$index_a], $cur_logic_dist);
                            push(@matched_cell_list,\@matched_cell_element);
                        }else{
                            my @matched_cell_element = ();
                            @matched_cell_element=("None_O", "None_O", "None_O",$ref_net,
                                                    $current_cell_name,$current_cell_refname, $current_pin_list[$index_a], $cur_logic_dist);
                            push(@matched_cell_list,\@matched_cell_element);
                        }
                    }

                }else{
                     croak ("$ref_net was not found in ref_netlink_hash \n");
                }
                $index_a=$index_a+1;
            }
        }
        else{
            croak ("(func_cells_finder): direction para: $direction illegal\n");
        }
    }#$cur_logic_dist
    return @matched_cell_list
}
