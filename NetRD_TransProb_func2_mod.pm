package NetRD_TransProb_func2_mod;
use strict;
use warnings FATAL => 'all';
# use Carp;
use Carp;
use feature qw(switch);
no warnings 'experimental::smartmatch';

use Storable qw(dclone);
use Verilog::Netlist;
#use Data::Dumper;
# Setup options so files can be found
use Verilog::Getopt;
use Time::HiRes qw( time );
use List::Util qw(first min max);
# Functional interface
# use Text::CSV_XS qw( csv );


use Exporter qw(import);
our @EXPORT_OK = qw(TransProb_caluer);


our @special_cell_refname=
        qw/dffacs1
         dffacs2
         dffascs1
         dffascs2
         dffass1
         dffass2
         dffcs1
         dffcs2
         dffles1
         dffles2
         dffs1
         dffs2
         dffscs1
         dffscs2
         dffss1
         dffss2
         sdffacs1
         sdffacs2
         sdffascs1
         sdffascs2
         sdffass1
         sdffass2
         sdffcs1
         sdffcs2
         sdffles1
         sdffles2
         sdffs1
         sdffs2
         sdffscs1
         sdffscs2
         sdffss1
         sdffss2
         lclks1
         lclks2
         lcs1
         lcs2
         lnnds1
         lnnds2
         lnors1
         lnors2
         lscs1
         lscs2
         lss1
         lss2 /;

# CLK CLRB DIN SETB EB RIN SIN SDIN SSEL
# our @special_cell_pin_name=
#         qw/D
#             SI
#             SE
#             CLK
#             GCLK
#             SN
#             RN
#             EN
#             RSTB
#             RETN
#             SETB/;

our @special_cell_pin_name=
        qw/DIN
           SDIN
           SSEL
           CLK
           SIN
           RIN
           EB
           CLRB
           SETB/;



# my $ref_Gate_HASH_ORI;
# my $ref_Netlink_HASH;
# my $ref_Design_input_Port;
# my $ref_Design_output_Port;

# my $my_folder_path="jqmul/comb_func";
# my $my_source_file_name="jqmul_trojan_inserted_reassign.v";
# my $my_lib_folder_dir="cell_lib";
# my $my_lib_file_name="saed90nm.v";
# my $my_output_path="/home/scyu/Documents/Pycharm_Projects/Word2vec_Netlist_Sample_modbatch/json_temp_file/jqmul/comb_func/jqmul_trojan_inserted_reassign";
# ($ref_Gate_HASH_ORI, $ref_Netlink_HASH, $ref_Design_input_Port,$ref_Design_output_Port)=TransProb_caluer(   $my_folder_path,
#                                                                                                             $my_source_file_name,
#                                                                                                             $my_lib_folder_dir,
#                                                                                                             $my_lib_file_name,
#                                                                                                             $my_output_path);


sub TransProb_caluer{
    my $folder_path=shift;
    my $source_file_name=shift;
    my $lib_folder_dir=shift;
    my $lib_file_name=shift;
    my $output_path=shift;



    my $source_library_directory="verilog/".$folder_path;
    my $lib_library_directory="verilog/".$lib_folder_dir;



    my $opt = new Verilog::Getopt;
    $opt->parameter( "+incdir+verilog",
                     "-y",$source_library_directory,
                     # "+incdir+verilog/cell_lib",
                     # "-v","verilog/cell_lib/saed90nm.v",
                     "+incdir+".$lib_library_directory,
                     "-v",$lib_library_directory."/".$lib_file_name,
                     );
    print '['.$_."]\n" for $opt->get_parameters;
    #printf("%-45s\n","1234567890");
    #printf("%s\n","1234567890");

    # Prepare netlist
    my $nl = new Verilog::Netlist(options => $opt,
                                  #link_read => 1,
                                  use_pinselects => 1,

                                 );
    foreach my $file ($source_file_name) {
        $nl->read_file(filename=>$file);
    }

    # Read in any sub-modules
    $nl->link();
    #$nl->lint();  # Optional, see docs; probably not wanted
    $nl->exit_if_error();
    #printf '['.$_."]\n" for $nl->files_sorted;
    foreach my $file ($nl->files_sorted) {
            printf("%s\n", $file->name);
    }


    ####################################################################
    # set start_time
    my @Design_input_Port=();
    my %INputPORT_HASH=();
    foreach my $mod ($nl->top_modules_sorted) {
        printf("Top module: %s\n", $mod->name);
        foreach my $port ($mod->ports_sorted) {
            # printf("all the ports: %sput %s\n", $port->direction, $port->name);
            my $ref_net=$port->net;

            #define the netname of this port, one-bit and  multi-bit
            my $netname = $ref_net->name;
            if(defined ($ref_net->msb) || defined ($ref_net->lsb)){
                my @sbits=($ref_net->msb,$ref_net->lsb);
                my $min_bit = min @sbits;
                my $max_bit = max @sbits;

                for my $num ($min_bit..$max_bit){
                    my $netname_temp=$netname."[".$num."]";
                    # printf("%s\n",$netname_temp);
                     if($port->direction eq "in") {
                         push(@Design_input_Port,$netname_temp);
                         $INputPORT_HASH{$netname_temp}=1;
                     }#if direction
                }#for #num
            }else{
               if($port->direction eq "in") {
                   # push(@Design_input_Port,$port->name);
                   push(@Design_input_Port,$netname);
                   $INputPORT_HASH{$netname}=1;
               }


            }

            # printf("protname:%s width:%d netname:%s netname_msb_lsb:%s\n",$port->name,$ref_net->width // 0, $ref_net->name, $netname);
            # croak( "error: port width >1" )     #fix
            #     if (defined ($ref_net->msb) || defined ($ref_net->lsb)); #fix

        }
    }
    # printf("input ports: @Design_inputPort\n");

    # my %net=();
    # $net{'Prob_0'}=-1;
    # $net{'prob_1'}=-1;

    my @Gate_Queue=();
    my @Design_output_Port=();
    my %wire=();
    my %Net_HASH=();
    foreach my $mod ($nl->top_modules_sorted) {
        foreach my $port ($mod->ports_sorted) {
            # printf("all the ports: %sput %s\n", $port->direction, $port->name);
            # print("1111zn") if defined $port->net->msb;
            my $ref_net=$port->net;
            my $netname = $ref_net->name;
            if(defined ($ref_net->msb) || defined ($ref_net->lsb)){
                my @sbits=($ref_net->msb,$ref_net->lsb);
                my $min_bit = min @sbits;
                my $max_bit = max @sbits;

                for my $num ($min_bit..$max_bit){
                    my $netname_temp=$netname."[".$num."]";
                    # printf("%s\n",$netname_temp);
                     if($port->direction eq "out") {
                         push(@Design_output_Port,$netname_temp);
                         $wire{'Prob_0'}= -1;
                         $wire{'Prob_1'}= -1;
                         $Net_HASH{$netname_temp}={%wire};
                     }#if direction
                }#for #num
            }else{
               if($port->direction eq "out") {
                   # push(@Design_input_Port,$port->name);
                   push(@Design_output_Port,$netname);
                   $wire{'Prob_0'}= -1;
                   $wire{'Prob_1'}= -1;
                   $Net_HASH{$netname}={%wire};
               }
            }
            # croak ("error: port width >1") #fix
            #     if (defined ($ref_net->msb) || defined ($ref_net->lsb)); #fix
            # if($port->direction eq "out") {
            #     push(@Design_output_Port,$port->name);
            #
            #     $wire{'Prob_0'}= -1;
            #     $wire{'Prob_1'}= -1;
            #     $Net_HASH{$port->name}={%wire};
            # }
        }
    }

    # foreach my $sig (keys %Net_HASH){
        # printf("key: %s, value: %s \n", $sig, $Net_HASH{$sig});
        # my %wire=%{$Net_HASH{$sig}};
        # printf("Prob_0 : %s \n", ${wire}{'Prob_0'});
        # printf("Prob_0 : %s \n", ${Net_HASH{$sig}}{'Prob_0'});
    # }

    # my @Design_Cells=();
    # foreach my $mod ($nl->top_modules_sorted) {
    #     foreach my $cell ($mod->cells_sorted) {
    #         push(@Design_Cells,$cell->name);
    #         }
    # }

    my %comp=();
    $comp{'cell_reference_name'}='None';
    $comp{'list_input_net'}='None';
    $comp{'list_input_pin'}='None';
    $comp{'no_inputs'}=-1;
    $comp{'list_output_net'}='None';
    $comp{'list_output_pin'}='None';
    $comp{'no_outputs'}=-1;
    $comp{'no_updated_inputs'}=-1;

    my %net=();
    # $net{'net_reference_name'}='None';
    # $net{'list_input_cell_pin'}=[];
    # $net{'no_inputs'}=-1;
    # $net{'list_output_cell_pin'}=[];
    # $net{'no_outputs'}=-1;

    # my @cell_pin_pair=();
    #(cellname,pinname)


    # foreach my $cell (@special_cell_refname){
    #     printf("%s\n", $cell);
    # }
    my @Design_Cells=();
    my %Gate_HASH=();
    my %Netlink_HASH=();
    foreach my $mod ($nl->top_modules_sorted) {
        foreach my $cell ($mod->cells_sorted) {

            push(@Design_Cells,$cell->name);
            # printf('\b1/q1/m0/c_reg[0]');
            printf("cell_name: %s\n",$cell->name);
            # if($cell->name eq '\b1/q1/m0/c_reg[0] '){
            #     printf("match!\n")
            # }
            $comp{'cell_reference_name'}=$cell->submodname;
            # printf("cell_reference_name: %s\n",$cell->submodname);
            my @list_input_net=();
            my @list_input_pin=();
            my @list_output_net=();
            my @list_output_pin=();
            my $submod=$cell->submod;
            foreach my $pin ($cell->pins_sorted) {
                # printf($indent."     .%s(%s)\n", $pin->name, $pin->netname);
                # my $co_net=$sig->net;
                # croak "error: port width >1"
                # if (defined ($co_net->msb) || defined ($co_net->lsb));
                my $ref_port=$pin->port;
                my @ref_net=$pin->pinselects;



                my $net_size=@ref_net;#rer_net is a array of the PinSelects (range of nets attached to the respective pin of a cell)
                croak( "error: multi-nets to one pin >1") #if there is more than one net connect to the pin, the netlist is illegal in script
                    if ($net_size > 1);

                my $net_name=$ref_net[0]->netname;
                if (defined ($ref_net[0]->msb) || defined ($ref_net[0]->lsb)){
                    if(abs($ref_net[0]->msb - $ref_net[0]->lsb )+1>1){
                        croak ("error: port width >1");
                    }
                    # printf("port:%s->pin:%s->net:%s\n",$ref_port->name,$pin->name,$ref_net[0]->bracketed_msb_lsb);
                    $net_name=$ref_net[0]->bracketed_msb_lsb
                }

                printf("port:%s->pin:%s->net:%s\n",$ref_port->name,$pin->name,$net_name);

                # printf("     .%s(%s)\n", $pin->name,$ref_net[0]->netname);
                if($ref_port->direction eq "in"){
                    push(@list_input_net,$net_name);
                    push(@list_input_pin,$pin->name);
                }
                if($ref_port->direction eq "out"){
                    push(@list_output_net,$net_name);
                    push(@list_output_pin,$pin->name);
                }
            }
            $comp{'list_input_net'}=[@list_input_net];
            $comp{'list_input_pin'}=[@list_input_pin];
            $comp{'no_inputs'}=@list_input_net;
            $comp{'list_output_net'}=[@list_output_net];
            $comp{'list_output_pin'}=[@list_output_pin];
            $comp{'no_outputs'}=@list_output_net;
            $comp{'no_updated_inputs'}=0;
            $Gate_HASH{$cell->name}={%comp};
    #################################################################################
            #additional codes for $Netlink_HASH
            my $index=0;
            foreach my $net_name (@list_input_net){
                my @cell_pin_pair=($cell->name,$comp{'cell_reference_name'},$list_input_pin[$index]);

                if(exists $Netlink_HASH{$net_name}){
                    push(@{$Netlink_HASH{$net_name}{'list_input_cell_pin'}},\@cell_pin_pair);
                }else{
                    $net{'list_input_cell_pin'}=[];
                    $net{'no_inputs'}=-1;
                    $net{'list_output_cell_pin'}=[];
                    $net{'no_outputs'}=-1;

                    push(@{$net{'list_input_cell_pin'}},\@cell_pin_pair);
                    $Netlink_HASH{$net_name}={%net};

                }
                $Netlink_HASH{$net_name}{'no_inputs'}=@{$Netlink_HASH{$net_name}{'list_input_cell_pin'}};
                $index++;
            }

            $index=0;
            foreach my $net_name (@list_output_net){
                my @cell_pin_pair=($cell->name,$comp{'cell_reference_name'},$list_output_pin[$index]);

                if(exists $Netlink_HASH{$net_name}){
                    push(@{$Netlink_HASH{$net_name}{'list_output_cell_pin'}},\@cell_pin_pair);
                }else{
                    $net{'list_input_cell_pin'}=[];
                    $net{'no_inputs'}=-1;
                    $net{'list_output_cell_pin'}=[];
                    $net{'no_outputs'}=-1;

                    push(@{$net{'list_output_cell_pin'}},\@cell_pin_pair);
                    $Netlink_HASH{$net_name}={%net};

                }
                $Netlink_HASH{$net_name}{'no_outputs'}=@{$Netlink_HASH{$net_name}{'list_output_cell_pin'}};
                $index++;

            }
    #################################################################################

            # my $submod=$cell->submod;
            # foreach my $sig ($submod->ports_sorted) {
            #     printf("      %sput %s\n", $sig->direction, );
            #     if($sig->direction eq "in"){
            #         push(@list_input_pin,$sig->name);
            #     }
            #     if($sig->direction eq "out"){
            #         push(@list_output_pin,$sig->name);
            #     }
            # }

            foreach my $in_net (@{$Gate_HASH{$cell->name}{'list_input_net'}}) {
                # printf("input_net %s\n",$in_net);
                # $in_net='Logic1';
                if(defined $INputPORT_HASH{$in_net}){
                    $wire{'Prob_0'}= 0.5;
                    $wire{'Prob_1'}= 0.5;
                    $Gate_HASH{$cell->name}{'no_updated_inputs'}++;
                    my $number_of_updated_inputs=$Gate_HASH{$cell->name}{'no_updated_inputs'};
                    my $number_of_input=$Gate_HASH{$cell->name}{'no_inputs'};
                    if($number_of_updated_inputs == $number_of_input){
                        my $my_cell_refname= $Gate_HASH{$cell->name}{'cell_reference_name'};
                        if(grep /^\Q$my_cell_refname\E$/, @special_cell_refname){
                            printf("match %s, do not add to gate queue\n",$my_cell_refname);
                        }else{
                            push(@Gate_Queue,$cell->name);
                        }
                    }
                }elsif( ($in_net eq 'Logic1') or ($in_net eq "1'b1") ){ #modified in TPSVM
                    $wire{'Prob_0'}= 0;
                    $wire{'Prob_1'}= 1;
                    $Gate_HASH{$cell->name}{'no_updated_inputs'}++;
                    my $number_of_updated_inputs=$Gate_HASH{$cell->name}{'no_updated_inputs'};
                    my $number_of_input=$Gate_HASH{$cell->name}{'no_inputs'};
                    if($number_of_updated_inputs == $number_of_input){
                        my $my_cell_refname= $Gate_HASH{$cell->name}{'cell_reference_name'};
                        if(grep /^\Q$my_cell_refname\E$/, @special_cell_refname){
                            printf("match %s, do not add to gate queue\n",$my_cell_refname);
                        }else{
                            push(@Gate_Queue,$cell->name);
                        }
                    }
                }elsif( ($in_net eq 'Logic0') or ($in_net eq "1'b0") ){ #modified in TPSVM
                    $wire{'Prob_0'}= 1;
                    $wire{'Prob_1'}= 0;
                    $Gate_HASH{$cell->name}{'no_updated_inputs'}++;
                    my $number_of_updated_inputs=$Gate_HASH{$cell->name}{'no_updated_inputs'};
                    my $number_of_input=$Gate_HASH{$cell->name}{'no_inputs'};
                    if($number_of_updated_inputs == $number_of_input){
                        my $my_cell_refname= $Gate_HASH{$cell->name}{'cell_reference_name'};
                        if(grep /^\Q$my_cell_refname\E$/, @special_cell_refname){ ###need regex fix?
                            printf("match %s, do not add to gate queue\n",$my_cell_refname);
                        }else{
                            push(@Gate_Queue,$cell->name);
                        }
                    }
                }else{
                    $wire{'Prob_0'}= -1;
                    $wire{'Prob_1'}= -1;
                }
                $Net_HASH{$in_net}={%wire};
            }
        }#each cell
    }#each mod
    my %Gate_HASH_ORI=%{dclone(\%Gate_HASH)};
    printf("####################\n");
    printf("STAGE 1 Passed\n");
    printf("####################\n");
    return(\%Gate_HASH_ORI, \%Netlink_HASH, \@Design_input_Port,\@Design_output_Port);

}

