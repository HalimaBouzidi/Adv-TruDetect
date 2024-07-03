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
# use Text::CSV_XS;

use File::Path qw(make_path);

use FindBin qw($Bin);
use lib "$Bin";

# use Perl_scripts::feature_functions qw(cellsblock_finder logic_loop_finder logic_SSC_finder sensitive_path_finder DominatorTree_builder);
use Getopt::Long;





my $start = time();


my $design_folder_dir='';
my $type_folder_dir='';
my $source_file_name='';
my $lib_folder_dir='';
my $lib_file_name='';




  GetOptions ("design_folder_path=s"   => \$design_folder_dir, # string
              "subtype_folder_path=s"  => \$type_folder_dir, # string
              "source_name=s"          => \$source_file_name,
              "lib_folder_path=s"        => \$lib_folder_dir,
              "lib_file_name=s"          => \$lib_file_name
  )   # string
  or die("Error in command line arguments\n");


if($lib_file_name eq 'lec25dscc25.v'){
    eval{require NetRD_TransProb_func2_mod;1;} or die "NetRD_TransProb_func2 not loaded!";
        NetRD_TransProb_func2_mod->import(qw(TransProb_caluer));


}else{
    printf("undefined cell library.");
}



my $ref_Gate_HASH_ORI;
my $ref_Netlink_HASH;
my $ref_Design_input_Port;
my $ref_Design_output_Port;

my $base_dir=$Bin;
my $sub_dir='json_temp_file';
my $sub_path = $base_dir.'/'.$sub_dir;
my $sub_design_folder_path = $sub_path.'/'.$design_folder_dir;
my $sub_subtype_folder_path = $sub_design_folder_path.'/'.$type_folder_dir;
my $sub_source_file_folder_path = $sub_subtype_folder_path.'/'.$source_file_name;

my @created = make_path($sub_source_file_folder_path);

# if (! -e $sub_path) {
#     mkdir($sub_path) or die "Can't create sub_dir, $!";}
#
# if (! -e  $sub_design_folder_path) {
#     mkdir($sub_design_folder_path) or die "Can't creat design_folder_path, $!";}
#
# if (! -e  $sub_subtype_folder_path) {
#     mkdir($sub_subtype_folder_path) or die "Can't creat subtype_folder_path , $!";}



# my $OUTput='\DFF_98/net323';
# my @list=('\DFF_98/net323' , '\DFF_175/net400' );
#
# my $match_times=grep(/^\Q$OUTput\E$/, @list);
# my $bar = "I am runoob site. welcome to runoob site.";
# if ($bar =~ /run/){
#    print "第一次匹配\n";
# }else{
#    print "第一次不匹配\n";
# }
#
# $bar = "run";
# if ($bar =~ /run/){
#    print "第二次匹配\n";
# }else{
#    print "第二次不匹配\n";
# }

printf("####################\n");
printf("RD&TP Stage: \n");
printf("Read Netlist and Gain Transition Probability\n");
printf("####################\n");
#will automatically find files in Verilog fold
my $combined_path=$design_folder_dir.'/'.$type_folder_dir;
($ref_Gate_HASH_ORI, $ref_Netlink_HASH, $ref_Design_input_Port,$ref_Design_output_Port)=TransProb_caluer($combined_path, $source_file_name.'.v', $lib_folder_dir, $lib_file_name, $sub_source_file_folder_path);

my %return_hash=( 'ref_Gate_HASH_ORI'=>$ref_Gate_HASH_ORI,
                  'ref_Netlink_HASH'=>$ref_Netlink_HASH,
                  'ref_Design_input_Port'=>$ref_Design_input_Port,
                  'ref_Design_output_Port'=>$ref_Design_output_Port
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
