
##Read a single Step One sheet

### Note the following UTF encoding

# >> u_char_name(916)
#[1] "GREEK CAPITAL LETTER DELTA"
# > u_char_name(1090)
# [1] "CYRILLIC SMALL LETTER TE"
# > 
#> u_char_name('U+00D1')
# [1] "LATIN CAPITAL LETTER N WITH TILDE"
# > u_char_name('U+00D1')
# [1] "LATIN CAPITAL LETTER N WITH TILDE"
# > u_char_name('U+00CE')
# [1] "LATIN CAPITAL LETTER I WITH CIRCUMFLEX"
# > 
# > intToUtf8(206)
# [1] "?"
# > intToUtf8(209)
# [1] "?"
# > 


stepone2df <- function(workbook,sheetname){
  require('gdata')
  my_sheet<- xls2csv(workbook,sheetname)        
  result_df <- read.csv(my_sheet,
                        na.strings=c("Undetermined","NA"),
                        skip=6,
                        colClasses=rep("character",25),
                        as.is=TRUE,
                        fileEncoding="UTF-8",
                        strip.white=TRUE,
                        header=TRUE)
  if (Sys.info()['sysname'] == "Windows") {
    
    tcode<- 209
    dcode <- 206
  } else {
    tcode<- 1090
    dcode <- 916
  }
  colnames(result_df)<- gsub(intToUtf8(dcode),'delta',colnames(result_df))
  colnames(result_df)<- gsub(intToUtf8(tcode),'t',colnames(result_df))
  colnames(result_df)<- gsub("\\.","",colnames(result_df))
  return(result_df[result_df$Task !="",])
}

## get correct task factor
get_stepone_task <- function(sample_str) {
  if (length(grep("IRC",sample_str) > 0)){
    return("IRC")
  } else if(sample_str=='NT') {
    return("NTC")
  } else {
    return("Unknown")
  }
}


