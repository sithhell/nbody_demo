# @ job_name = variance_strong
# @ comment = "My Second Job"
# @ error  = $(job_name).$(jobid).out
# @ output = $(job_name).$(jobid).out
# @ environment = COPY_ALL
# @ wall_clock_limit = 00:10:00
# @ notification = error
# @ notify_user = gentryx@gmx.de
# @ job_type = bluegene
# @ bg_size = 1
# @ queue
echo "go go, powerrangers!"
runjob --exe main_strong --args "-t 1" --ranks-per-node 1
echo "done"
