for i in {1..19}; do
    netconvert -s all.net.xml -i ../traffic_light/"$i".tll.xml -o "$i".net.xml
done
