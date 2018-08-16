#for f in *.mp4; do echo $f; DATE=$(echo $f | cut -d' ' -f 1); echo "the date is
#$DATE"; if (($DATE < 20170501)); then echo "$DATE passes the cutoff of 20170501"; echo "copying $f to .."; cp "$f"
#..; fi; done;

for f in *
do
echo $f
DATE=$(echo $f | cut -d' ' -f 1)
echo "the date is $DATE"
if (($DATE < 20170501))
then
echo "$DATE passes the cutoff of 20170501"
echo "copying $f to .."
cp "$f" ..
fi
done

