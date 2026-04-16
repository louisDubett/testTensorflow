#!/bin/bash -x
for f in *.NEF; do

FMT="NEF"
FILE=$(basename $f .$FMT)
DIR=$(dirname $f)
ISO=$(exiftool "$DIR/$FILE.$FMT" -t -ISO | cut -f2)
DEST_FILE=$DIR/$FILE.raw.jpg

echo "$FILE.$FMT :: Using ISO $ISO"

dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
  convert - -verbose \
    -quality '90' -auto-orient \
  $DEST_FILE

exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE

#rotate
convert $DEST_FILE -rotate 5.0 $DIR/$FILE.raw-rotate-plus5.jpg
convert $DEST_FILE -rotate -5.0 $DIR/$FILE.raw-rotate-minus5.jpg

DEST_FILE=$DIR/$FILE.denoise.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
  convert - -verbose \
-wavelet-denoise $ISO \
    -quality '90' -auto-orient \
  $DEST_FILE

exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE

DEST_FILE=$DIR/$FILE.level.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
  convert - -verbose \
    -auto-level \
    -quality '90' -auto-orient \
  $DEST_FILE
exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE


DEST_FILE=$DIR/$FILE.grey.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
  convert - -verbose \
    -grayscale Rec709Luminance  \
    -quality '90' -auto-orient \
  $DEST_FILE
exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE

DEST_FILE=$DIR/$FILE.mean-shift.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
  convert - -verbose \
   -mean-shift 7x7+10% \
    -quality '90' -auto-orient \
  $DEST_FILE
exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE

DEST_FILE=$DIR/$FILE.more-brightness.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
  convert - -verbose \
   -set option:modulate:colorspace hsb -modulate 105,95  \
    -quality '90' -auto-orient \
  $DEST_FILE
exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE

DEST_FILE=$DIR/$FILE.less-brightness.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
  convert - -verbose \
   -set option:modulate:colorspace hsb -modulate 90,100  \
    -quality '90' -auto-orient \
  $DEST_FILE
exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE


DEST_FILE=$DIR/$FILE.sharpen.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
  convert - -verbose \
   -sharpen '2x2.0' \
    -quality '90' -auto-orient \
  $DEST_FILE
exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE

DEST_FILE=$DIR/$FILE.sigmoidal.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
   -sigmoidal-contrast '10,50%'  \
    -quality '90' -auto-orient \
  $DEST_FILE
exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE

DEST_FILE=$DIR/$FILE.level.jpg
dcraw -w -c -v -b ${2:-1} -H 0 -T "$DIR/$FILE.$FMT" | \
    -level '0%,100%,1.2' \
    -quality '90' -auto-orient \
  $DEST_FILE
exiftool -overwrite_original -TagsFromFile $DIR/$FILE.$FMT $DEST_FILE


done
