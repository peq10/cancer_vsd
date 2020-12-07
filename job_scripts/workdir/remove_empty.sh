find . -empty -exec rm "{}" \;
echo "Number successful"
find . -name "*.sh.o*"  -exec grep -o "Finished successfully" {} \; | wc -l
find . -name "*.sh.o*"  -exec grep -l "Finished successfully" {} \; -exec rm "{}" \;
