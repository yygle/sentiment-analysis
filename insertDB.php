<?php

$movname = @$_POST['movname'];
$nickname = @$_POST['username'];
$comment = @$_POST['comments'];

//echo $movname.'<br>';
//echo $nickname.'<br>';
echo $nickname.', your review : <br>'.$comment.'<br>';




$path = writeCommand2Txt($comment,$movname);


$output = executePython('test_one_sentence.py', $path);

insertDB('localhost','root','',$movname,$nickname,$comment, $output);
print($output);

function insertDB($server, $username, $PW,$movname,$nickname,$comment,$label)
{
    
 //   $link =mysql_connect('10.61.6.103', 'root','ransupport');
    $link =mysqli_connect($server, $username);
   if (!$link) die('Could no connect: ' . mysql_error());
    
    $sql = "INSERT INTO comp7404.Movie_Review VALUES ('1', NOW(), '".$movname."', '".$nickname."','".$comment."','".$label."');";
    $result = mysqli_query($link,$sql);
    if(!$result)echo 'Couldnt insert : <br>'.$sql.'<br>' .mysqli_error();
    
    
}



function executePython($filename,$commentPath)
{
    //$command = escapeshellcmd('python '.$filename." 2>&1");
    $command = 'python '.$filename." ".$commentPath." 2>&1";
   // $command = escapeshellcmd('ls');
    //$output = shell_exec($command);
    $output = shell_exec($command);
    if(!isset($output)) $output = "null";

    return $output;
}


function writeCommand2Txt($txt,$movname)
{
    $date = new DateTime();
    $time = $date->format('YmdHis');
    if(!isset($movname)|| $movname == "") $movname = "temp";
    
    
    $dir = "./output/" . $movname . "/";

    if (!file_exists($dir)) {
    mkdir($dir, 0777);
    }
        
    $filePath = "./output/".$movname."/comment".$time.".txt";
    $filePath2="./comment.txt";
    //if(file_exists($filePath)) unlink($filePath);
    file_put_contents($filePath,$txt);  
    file_put_contents($filePath2,$txt);    
    //$myfile = fopen($filePath, "w") or die("Unable to open file!");
    //fwrite($myfile, $txt);
    //fclose($myfile);
    return $filePath2;
    
}




?>