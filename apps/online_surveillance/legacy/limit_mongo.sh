#!/bin/bash
# Script to limit mongoDB memory usage.
# Author: TechPaste.Com
############################################

export _arg="$1"
if [ -z "$_arg" ]
then
echo "Memory to Allocate is empty"
echo "Usage: ./script.sh 1536"
echo "Note: Here 1536 is 1536 MB of RAM"
exit 1;

fi

_check_tools() {
_cmd=$1
command -v $_cmd >/dev/null 2>&1 || { echo >&2 "Script requires $_cmd but it's not in path or not installed. Aborting."; exit 1; }

}
_pre_setup() {

_check_tools ps
_check_tools pmap
_check_tools cgcreate
_check_tools sync
_check_tools cgclassify
_check_tools tail
_check_tools free

memaloc=`echo "$(($_arg * 1024 * 1024))"`
mongopid=`ps -eaf | grep mongod |grep -v grep | awk -F" " '{print $2}'`
if [[ $mongopid == "" ]]; then

echo " Mongo DB is not running. Please start the Mongo DB service first.";
echo "Example start command: mongod --fork --dbpath /data/db/ --logpath /opt/mongodb/mongodb.log"
exit 1;

fi
echo
echo "Mongo DB Process: "
echo
echo "########################################################"
ps -eaf | grep mongod |grep -v grep
echo "########################################################"
echo "Current MongoDB RAM usage:"
echo "$(( `pmap -x $mongopid | tail -1 | awk -F" " '{print $3}'` ))KB= $(( `pmap -x $mongopid | tail -1 | awk -F" " '{print $3}'` / 1024 ))MB"
echo "########################################################"
echo
}
_mem_setup() {
echo
echo "1. Creating control group :MongoLimitGroup."
echo
echo "Running cgcreate -g memory:MongoLimitGroup"
cgcreate -g memory:MongoLimitGroup
echo
echo "2. Specifying $memaloc bytes memory needs to be allocated for this group"
echo
echo "echo $memaloc > /sys/fs/cgroup/memory/MongoLimitGroup/memory.limit_in_bytes"
echo $memaloc > /sys/fs/cgroup/memory/MongoLimitGroup/memory.limit_in_bytes
echo
echo "3. Dropping pages already stayed in cache..."
echo
echo "sync; echo 3 > /proc/sys/vm/drop_caches"
sync; echo 3 > /proc/sys/vm/drop_caches
echo
echo "4. Assigning a server to be created for control group"
echo
echo "cgclassify -g memory:MongoLimitGroup $mongopid"
cgclassify -g memory:MongoLimitGroup $mongopid
echo
echo "########################################################"
echo "Post Setup MongoDB RAM usage:"
echo "$(( `pmap -x $mongopid | tail -1 | awk -F" " '{print $3}'` ))KB= $(( `pmap -x $mongopid | tail -1 | awk -F" " '{print $3}'` / 1024 ))MB"
echo "########################################################"
}

Main() {
_pre_setup
_mem_setup
}
Main